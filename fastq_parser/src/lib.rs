#![feature(test)]
extern crate test;
use bio::io::fastq::{self, Record};
use bloomfilter::Bloom;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::HashMap;

pub mod functionalities {
    use std::str;

    use bio::io::fastq::{self, Record};
    use rayon::prelude::*;
    pub struct Student {
        pub age: u32,
    }
    pub fn parse_fastq_file(file_path: String) -> Result<Vec<String>, String> {
        let reader = fastq::Reader::from_file(file_path).unwrap();
        reader
            .records()
            .into_iter()
            .map(|record| {
                let record = match record {
                    Ok(record) => record,
                    Err(_) => return Err("invalid record!".to_string()),
                };
                let record_bytes = record.seq();
                let string_read = match str::from_utf8(record_bytes) {
                    Ok(str) => str.to_string(),
                    Err(_) => return Err("Error converting bytes into utf-8 string".to_string()),
                };
                Ok(string_read)
            })
            .collect()
    }
    //fn extract_kmers() -> Vec<String> {}
    pub fn parse_fastq_file_parallel(file_path: String) -> Result<Vec<String>, String> {
        let reader = fastq::Reader::from_file(file_path).unwrap();
        let records: Vec<Result<Record, _>> = reader.records().collect();

         records
            .into_par_iter()
            .map(|record| {
                let record = match record{
                    Ok(record) => record,
                    Err(_) => return Err("Invalid fastq record".to_string())
                };
                let string_read = match str::from_utf8(record.seq()){
                    Ok(string_read) => string_read,
                    Err(_) => return Err("error converting byte array into string".to_string())
                };
                Ok(string_read.to_string()) 
            })
            .collect()
    }
    pub fn parallel_increment(arr: &mut Vec<Student>) -> Result<(), String> {
        let incremented: Vec<u32> = arr.par_iter_mut().map(|student| student.age + 1).collect();
        Ok(())
    }
}
#[pymodule]
fn fastq_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(extract_kmers, m)?)?;
    Ok(())
}

#[pyfunction]
fn extract_kmers(file_path: String, kmer_length: usize) -> PyResult<Vec<String>> {
    let reads = match generate_string_reads(file_path) {
        Ok(reads) => reads,
        Err(_) => {
            return Err(PyErr::new::<PyTypeError, _>("Something went wrong"));
        }
    };

    let mut bloom = match Bloom::new_for_fp_rate(1000000, 0.001) {
        Ok(bloom) => bloom,
        Err(_) => {
            return Err(PyErr::new::<PyTypeError, _>(
                "Something went wrong initiating bloom filter",
            ));
        }
    };
    let mut hash_map = HashMap::new();

    reads.iter().for_each(|read| {
        let kmers = generate_kmers(read, &kmer_length);
        kmers.iter().for_each(|kmer| {
            if bloom.check(kmer) {
                hash_map
                    .entry(kmer.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(0);
            } else {
                bloom.set(kmer);
            }
        });
    });

    //let occurences:Vec<i32> = hash_map.clone().into_values().collect();
    let kmers: Vec<String> = hash_map.into_keys().collect();
    Ok(reads)
}

fn generate_kmers(read: &str, kmer_length: &usize) -> Vec<String> {
    read.chars()
        .collect::<Vec<char>>()
        .windows(*kmer_length)
        .map(|x| x.iter().collect::<String>())
        .collect()
}
fn generate_string_reads(file_path: String) -> Result<Vec<String>, &'static str> {
    match fastq::Reader::from_file(file_path) {
        Ok(reader) => {
            let mut read_vector = vec![];
            for result in reader.records() {
                if let Ok(record) = result {
                    let string_record = byte_to_string(record.seq()).unwrap();
                    read_vector.push(string_record);
                }
            }
            return Ok(read_vector);
        }
        Err(_) => {
            return Err("Something went wrong");
        }
    }
}

#[pyfunction]
fn parse_fastq_file_parallel(file_path: String) -> PyResult<Vec<String>> {
    let mut reader = match fastq::Reader::from_file(file_path) {
        Ok(reader) => reader,
        Err(_) => return Err(PyErr::new::<PyTypeError, _>("error in reading file")),
    };

    let records_list: Vec<Result<Record, _>> = reader.records().collect();
    records_list.into_par_iter().map(|record| {
        let record = match record {
            Ok(record)  => record,
            Err(e) => return Err(PyErr::new::<PyTypeError, _>("Something went wrong")),
        };
        let string_read = str::from_utf8(record.seq()).unwrap();
        Ok(string_read.to_string())
    }).collect()

}
//creates python bindings that will be used for parsing fastq files
#[pyfunction]
fn parse_fastq_file(file_path: String) -> PyResult<Vec<String>> {
    match generate_string_reads(file_path) {
        Ok(result) => Ok(result),
        Err(_) => Err(PyErr::new::<PyTypeError, _>("Something went wrong")),
    }
}
#[pyfunction]
fn write_fastq_file(
    dst_filename: String,
    src_filename: String,
    matrix: &Bound<'_, PyArray2<u8>>,
) -> PyResult<()> {
    let mut writer = match fastq::Writer::to_file(dst_filename) {
        Ok(writer) => writer,
        Err(_) => {
            return Err(PyErr::new::<PyTypeError, _>("Invalid bytes array"));
        }
    };

    let reader = match fastq::Reader::from_file(src_filename) {
        Ok(reader) => reader,
        Err(_) => {
            return Err(PyErr::new::<PyTypeError, _>(
                "Destination file name cannot be located",
            ));
        }
    };

    let np_matrix = unsafe { matrix.as_array() };
    let result: Result<Vec<String>, _> = np_matrix
        .rows()
        .into_iter()
        .map(|row| {
            let vector_row = row.to_vec();
            byte_to_string(&vector_row)
        })
        .collect();

    match result {
        Ok(rows_as_strings) => {
            for (record, row) in reader.records().zip(rows_as_strings.iter()) {
                if let Ok(record) = record {
                    let new_record = Record::with_attrs(
                        record.id(),
                        record.desc(),
                        row.as_bytes(),
                        record.qual(),
                    );

                    if let Err(_) = writer.write_record(&new_record) {
                        return Err(PyErr::new::<PyTypeError, _>(
                            "Error writing new record to the file",
                        ));
                    }
                } else {
                    return Err(PyErr::new::<PyTypeError, _>(
                        "Something went wrong during extracting record",
                    ));
                }
            }
        }
        Err(e) => {
            return Err(e);
        }
    }
    Ok(())
}

//this function unwraps for now. Add better error handling
fn byte_to_string(byte_array: &[u8]) -> Result<String, PyErr> {
    match std::str::from_utf8(byte_array) {
        Ok(string) => Ok(string.to_string()),
        Err(_) => Err(PyErr::new::<PyTypeError, _>("Invalid bytes array")),
    }
}

#[cfg(test)]
mod tests {
    use crate::functionalities::Student;

    use super::functionalities;
    use rand::Rng;
    use test::{black_box, Bencher};

    #[bench]
    fn parse_parallel(b: &mut Bencher) {
        //add file here
        let fastq_filepath =
            "/home/g6i1o0/Documents/dask-cuda/genetic-assets/ecoli_30x_3perc_single.fq".to_string();
        b.iter(|| {
            black_box(functionalities::parse_fastq_file_parallel(
                fastq_filepath.clone(),
            ))
        });
    }
    #[bench]
    fn parse_serial(b: &mut Bencher) {
        //add file here
        let fastq_filepath =
            "/home/g6i1o0/Documents/dask-cuda/genetic-assets/ecoli_30x_3perc_single.fq".to_string();
        b.iter(|| black_box(functionalities::parse_fastq_file(fastq_filepath.clone())));
    }
    #[bench]
    fn increment_parallel(b: &mut Bencher) {
        //add file here
        let mut rng = rand::rng();
        let mut students: Vec<Student> = (0..200_000)
            .map(|_| Student {
                age: rng.random_range(0..100),
            })
            .collect();
        b.iter(|| black_box(functionalities::parallel_increment(&mut students)));
    }
}
