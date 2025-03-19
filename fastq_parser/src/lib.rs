#![feature(test)]
extern crate test;
use std::fs::File;
use std::error::Error;
use std::io::{BufReader, SeekFrom, Seek};
use bio::io::fastq::Record;
use bloomfilter::Bloom;
use numpy::ndarray::{Array2, ArrayD, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::result;
use seq_io::fastq::Reader;
//use seq_io::parallel::read_parallel;
use fastq::{Parser, Record as FastqRecord};
use std::collections::hash_map::HashMap;
use std::str;
use std::sync::{Arc, Mutex};
static NTHREADS:usize = 48;
#[pymodule]
fn fastq_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_parse_fastq, m)?)?;
    //m.add_function(wrap_pyfunction!(parse_fastq_file_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_fastq_foreach, m)?)?;
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
//#[pyfunction]
//fn parallel_write_fastq()

#[pyfunction]
fn parse_fastq_foreach(file_path:String, start_offset:u64, batch_size:u64) -> PyResult<Vec<String>>{

    let file = File::open(file_path).unwrap();
    let mut reader = BufReader::new(file);
    reader.seek(SeekFrom::Start(start_offset)).unwrap();
    let parser = Parser::new(reader);
    let mut result = vec![];
    let mut count = 0;
    parser.each(|record|{
        if count  > batch_size {
            return false;
        }
        let byte_seq = record.seq();
        let string_seq = str::from_utf8(byte_seq).unwrap().to_string();
        result.push(string_seq);
        count += 1;
        return true;
    }).expect("Error fastq file");
    Ok(result)

}
#[pyfunction]
fn parallel_parse_fastq(file_path:String) -> PyResult<Vec<String>>{
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let parser = Parser::new(reader);
    //this might be slow wrapping in arc mutex 
    let global_results = Arc::new(Mutex::new(vec![]));


    let collected_results:Result<Vec<()>, _> = parser.parallel_each(16,
        {
            //the worker closure that will be executed on each worker threads
            let mut results = Arc::clone(&global_results);
            move |records_sets|{
                let mut local_results = vec![];
                for record_set in records_sets{
                    for record in record_set.iter(){
                        let byte_seq = record.seq();
                        let string_seq = str::from_utf8(byte_seq).unwrap().to_string();
                        local_results.push(string_seq);
                    }
                }
                //append the local result into the global result that is wrapped by arc mutex
                //mutex to prevent race conditionz
                let mut global_results = results.lock().unwrap();
                global_results.extend(local_results);
            }
        }
    );
    //check any error happening within the threads
    collected_results?;

    //unwrap the inner data wrapped within arc mutex
    let inner = Arc::try_unwrap(global_results).unwrap().into_inner().unwrap();
    Ok(inner)
}
fn generate_kmers(read: &str, kmer_length: &usize) -> Vec<String> {
    read.chars()
        .collect::<Vec<char>>()
        .windows(*kmer_length)
        .map(|x| x.iter().collect::<String>())
        .collect()
}
fn generate_string_reads(file_path: String) -> Result<Vec<String>, &'static str> {
    match Reader::from_path(file_path) {
        Ok(mut reader) => {
            let mut read_vector = vec![];
            for result in reader.records() {
                if let Ok(record) = result {
                    let mut vec_record = record.seq.to_vec();
                    let string_record = byte_to_string(&mut vec_record).unwrap();
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

//creates python bindings that will be used for parsing fastq files
#[pyfunction]
fn parse_fastq_file(file_path: String) -> PyResult<Vec<String>> {
    let res = match generate_string_reads(file_path) {
        Ok(result) => result,
        Err(_) => return Err(PyErr::new::<PyTypeError, _>("Something went wrong")),
    };
    Ok(res)
}
#[pyfunction]
fn write_fastq_file(
    dst_filename: String,
    src_filename: String,
    matrix: &Bound<'_, PyArray2<u8>>,
) -> PyResult<()> {
    let mut writer = match bio::io::fastq::Writer::to_file(dst_filename) {
        Ok(writer) => writer,
        Err(_) => {
            return Err(PyErr::new::<PyTypeError, _>("Invalid bytes array"));
        }
    };

    let reader = match bio::io::fastq::Reader::from_file(src_filename) {
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
            let mut vector_row = row.to_vec();
            byte_to_string(&mut vector_row)
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
fn byte_to_string(byte_array: &mut Vec<u8>) -> Result<String, PyErr> {
    if let Some(cleaned) = byte_array.iter().position(|&b| b == 0) {
        byte_array.truncate(cleaned);
    }
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
