#![feature(test)]
extern crate test;
use bio::io::fastq::{self, Record};
use bloomfilter::Bloom;
use numpy::ndarray::{Array2, ArrayD, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use ofilter::SyncBloom;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::HashMap;
use std::str;

static NTHREADS: u32 = 48;
pub mod functionalities {
    use bio::io::fastq::{self, Record};
    use rand;
    use rayon::prelude::*;
    use std::{collections::HashMap, str};
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
    pub fn parse_fastq_file_parallel(
        file_path: String,
        kmer_length: usize,
    ) -> Result<Vec<Vec<String>>, String> {
        let reader = fastq::Reader::from_file(file_path).unwrap();
        //let mut bloom = bloomfilter::Bloom::new_for_fp_rate(1_000_000, 0.001).unwrap();
        //let mut hash_map = HashMap::new();

        let records: Vec<Result<Record, _>> = reader.records().collect();
        records
            .into_par_iter()
            .map(|record| {
                let record = match record {
                    Ok(record) => record,
                    Err(_) => return Err("Invalid fastq record".to_string()),
                };
                let string_read = match String::from_utf8(record.seq().to_vec()) {
                    Ok(string_read) => string_read,
                    Err(_) => return Err("error converting byte array into string".to_string()),
                };

                //we will be using bloom filter for this process
                let kmers = string_read
                    .chars()
                    .collect::<Vec<char>>()
                    .windows(kmer_length)
                    .map(|x| x.iter().collect::<String>())
                    .collect();

                //.for_each(|kmer| {
                //    //use bloom filter and use hashmap to store this
                //    hash_map.entry(kmer).and_modify(|val| *val + 1).or_insert(0);
                //});
                Ok(kmers)
            })
            .collect()
    }
    pub fn random_computation(age: &u32) {
        let mut rng = rand::rng();

        (0..20_000).for_each(|val| {
            val * age;
        });
    }
    pub fn parallel_increment(arr: &mut Vec<Student>) -> Result<usize, String> {
        Ok(arr
            .par_iter_mut()
            .map(|student| {
                random_computation(&student.age);
                student.age * 2
            })
            .count())
    }
    pub fn serial_increment(arr: &mut Vec<Student>) -> Result<usize, String> {
        Ok(arr
            .iter()
            .map(|student| {
                random_computation(&student.age);
                student.age * 2
            })
            .count())
    }
}
#[pymodule]
fn fastq_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_fastq_file, m)?)?;
    //m.add_function(wrap_pyfunction!(parse_fastq_file_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(extract_kmers, m)?)?;
    Ok(())
}

#[pyclass]
struct ParseResult {
    reads: Vec<String>,
    kmers: Vec<Vec<String>>,
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

//#[pyfunction]
//fn parse_fastq_file_parallel<'py>(
//    py: Python<'py>,
//    file_path: String,
//    kmer_length: usize,
//) -> Bound<'py, PyArray2<u8>> {
//    let reader = match fastq::Reader::from_file(file_path) {
//        Ok(reader) => reader,
//        Err(_) => panic!("error in reading file"),
//    };
//
//    let bloom: SyncBloom<String> = SyncBloom::new(1_000_000);
//    let mut hash_map: HashMap<String, u32> = HashMap::new();
//
//    let records_list: Vec<Result<Record, _>> = reader.records().collect();
//
//    let string_reads: Vec<&[u8]> = records_list
//        .into_par_iter()
//        .map(|record| {
//            let record = match record {
//                Ok(record) => record,
//                Err(_) => panic!("invalid record"),
//            };
//
//            record.seq()
//            //let string_read = str::from_utf8(record.seq()).unwrap();
//            //string_read.to_string()
//        })
//        .collect();
//    let utf_reads: Vec<Vec<u8>> = string_reads
//        .par_iter()
//        .map(|read| {
//            //convert chars into utf-u8 format
//            read.chars().map(|char| char as u8).collect()
//        })
//        .collect();
//    let kmers: Vec<Vec<String>> = string_reads
//        .par_iter()
//        .map(|read_string| {
//            let kmers: Vec<String> = read_string
//                .chars()
//                .collect::<Vec<char>>()
//                .windows(kmer_length)
//                .map(|x| x.iter().collect::<String>())
//                .collect();
//
//            kmers.iter().for_each(|kmer| {
//                //use bloom filter and check whether it exists
//            });
//            return kmers;
//        })
//        .collect();
//    let flatten: Vec<u8> = utf_reads.into_par_iter().flatten().collect();
//    Array2::from_shape_vec((string_reads.len(), 100), flatten)
//        .unwrap()
//        .into_pyarray(py)
//}
//creates python bindings that will be used for parsing fastq files
#[pyfunction]
fn parse_fastq_file(file_path: String) -> PyResult<Vec<String>> {
    match generate_string_reads(file_path) {
        Ok(result) => Ok(result),
        Err(_) => Err(PyErr::new::<PyTypeError, _>("Something went wrong")),
    }
}
//#[pyfunction]
//fn parse_fastq_file_v2_parallel(file_path: String) -> PyResult<Vec<String>> {
//    let mut reader = fastq::Reader::from_file(file_path).unwrap();
//    let records = match reader.records() {
//        Ok(records) => records,
//        Err(_) => return Err(PyErr::new::<PyTypeError, _>("Something went wrong")),
//    };
//    //let string_reads = records.
//}
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

    //#[bench]
    //fn parse_parallel(b: &mut Bencher) {
    //    //add file here
    //    let fastq_filepath =
    //        "/home/g6i1o0/Documents/dask-cuda/genetic-assets/ERR022075_1.fastq".to_string();
    //    b.iter(|| {
    //        black_box(functionalities::parse_fastq_file_parallel(
    //            fastq_filepath.clone(),
    //            19,
    //        ))
    //    });
    //}
    //#[bench]
    //fn parse_serial(b: &mut Bencher) {
    //    //add file here
    //    let fastq_filepath =
    //        "/home/g6i1o0/Documents/dask-cuda/genetic-assets/ERR022075_1.fastq".to_string();
    //    b.iter(|| black_box(functionalities::parse_fastq_file(fastq_filepath.clone())));
    //}
    #[bench]
    fn increment_serial(b: &mut Bencher) {
        let mut rng = rand::rng();
        let mut students: Vec<Student> = (0..20_000_000)
            .map(|_| Student {
                age: rng.random_range(0..100),
            })
            .collect();
        b.iter(|| black_box(functionalities::serial_increment(&mut students)));
    }
    #[bench]
    fn increment_parallel(b: &mut Bencher) {
        //add file here
        let mut rng = rand::rng();
        let mut students: Vec<Student> = (0..20_000_000)
            .map(|_| Student {
                age: rng.random_range(0..100),
            })
            .collect();
        b.iter(|| black_box(functionalities::parallel_increment(&mut students)));
    }
}
