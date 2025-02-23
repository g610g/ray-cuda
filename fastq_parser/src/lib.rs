use std::collections::hash_map::HashMap;
use bio::io::fastq::{self, Record};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods};
use bloomfilter::Bloom;

#[pymodule]
fn fastq_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(extract_kmers, m)?)?;
    Ok(())
}

#[pyfunction]
fn extract_kmers(file_path:String, kmer_length: usize) -> PyResult<Vec<String>> {
    let reads = match generate_string_reads(file_path){
        Ok(reads) => reads,
        Err(_) => {
            return Err(PyErr::new::<PyTypeError, _>("Something went wrong"));
        }
    };

    let mut bloom = match Bloom::new_for_fp_rate(1000000, 0.001 ){
        Ok(bloom) => bloom,
        Err(_) => {return Err(PyErr::new::<PyTypeError, _>("Something went wrong initiating bloom filter"));
        }
    };
    let mut hash_map = HashMap::new();

    reads.iter().for_each(|read|{
        let kmers = generate_kmers(read, &kmer_length);
        kmers.iter().for_each(|kmer|{
            if bloom.check(kmer){
                hash_map.entry(kmer.clone()).and_modify(|count| *count += 1 ).or_insert(0);
            }else{
                bloom.set(kmer);
            }
        });
    });

    //let occurences:Vec<i32> = hash_map.clone().into_values().collect();
    let kmers:Vec<String> = hash_map.into_keys().collect();
    Ok(reads)
}

fn generate_kmers(read:&str, kmer_length:&usize) -> Vec<String>{
    read.chars()
    .collect::<Vec<char>>()
    .windows(*kmer_length)
    .map(|x| x.iter().collect::<String>())
    .collect()
}
fn generate_string_reads(file_path:String) -> Result<Vec<String>, &'static str>{
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

//creates python bindings that will be used for parsing fastq files
#[pyfunction]
fn parse_fastq_file(file_path: String) -> PyResult<Vec<String>> {

    match generate_string_reads(file_path) {
        Ok(result) => Ok(result),
        Err(_) => Err(PyErr::new::<PyTypeError, _>("Something went wrong"))
    }
}
#[pyfunction]
fn write_fastq_file(dst_filename:String, src_filename:String, matrix:&Bound<'_, PyArray2<u8>>) -> PyResult<()>{
    let mut writer = match fastq::Writer::to_file(dst_filename){
        Ok(writer) => writer,
        Err(_)  => {
            return Err(PyErr::new::<PyTypeError, _>("Invalid bytes array"));
        }
    };

    let reader = match fastq::Reader::from_file(src_filename) {
        Ok(reader) => reader,
        Err(_) => {
            return Err(PyErr::new::<PyTypeError, _>("Destination file name cannot be located"));
        }
    };

    let np_matrix = unsafe {
        matrix.as_array()
    };
    let result :Result<Vec<String>, _> = np_matrix
            .rows()
            .into_iter().map(|row|{
                let vector_row = row.to_vec();
                byte_to_string(&vector_row)
        }).collect();

    match result {
        Ok(rows_as_strings) => {
            for (record, row) in reader.records().zip(rows_as_strings.iter()){

                if let Ok(record) = record {

                    let new_record =  Record::with_attrs(record.id(), record.desc(), row.as_bytes(),record.qual());

                    if let Err(_) = writer.write_record(&new_record) {
                        return Err(PyErr::new::<PyTypeError, _>("Error writing new record to the file"));
                    }
                }
                else {
                    return Err(PyErr::new::<PyTypeError, _>("Something went wrong during extracting record"));
                }
            }
        },
        Err(e) => {
            return Err(e);
        }
    }
    Ok(())
}

//this function unwraps for now. Add better error handling
fn byte_to_string(byte_array: &[u8]) -> Result<String, PyErr>{
    match std::str::from_utf8(byte_array){
        Ok(string) => Ok(string.to_string()),
        Err(_) => Err(PyErr::new::<PyTypeError, _>("Invalid bytes array"))

    }
}
