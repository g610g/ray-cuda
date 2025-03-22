#![feature(test)]
extern crate test;
use std::fs::{File, OpenOptions};
use anyhow::Result;
use std::io::{BufReader, BufWriter, Seek, SeekFrom};
use bio::io::fastq::Record;
use bloomfilter::Bloom;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use seq_io::fastq::{OwnedRecord, Reader as FastqReader, Record as RecordTrait, RecordSet, RefRecord};
//use seq_io::parallel::read_parallel;
use fastq::{Record as FastqRecord};
use seq_io::parallel::{parallel_fastq, read_parallel, Reader};
use seq_io_parallel::{ParallelProcessor, ParallelReader};
use std::collections::hash_map::HashMap;
use std::str::{self, from_utf8};
use std::sync::{Arc, Mutex};
use std::io::Write;
static NTHREADS:usize = 48;

#[derive(Clone)]
pub struct WriteCalculation {
    corrected_2d_reads: Vec<String>,
    writer: Arc<Mutex<Box<dyn Write + Send>>>
}

impl ParallelProcessor for WriteCalculation{
    fn process_record<'a, Rf: seq_io_parallel::MinimalRefRecord<'a>>(&mut self, record: Rf) -> Result<()> {
        let mut writer: std::sync::MutexGuard<'_, Box<dyn Write + Send>> = self.writer.lock().unwrap();
        let fastq_header = match str::from_utf8(record.ref_head()) {
            Ok(str_id) => str_id,
            Err(_) => {
                eprintln!("Invalid UTF-8 in record header");
                return Ok(()); // Return early if the header is invalid
            }
        };
        let id_number: usize = extract_id_number(fastq_header).unwrap();
        let corrected_sequence: &String = self.corrected_2d_reads.get((id_number - 1)  ).unwrap();
        seq_io::fastq::write_to(&mut *writer, record.ref_head(), corrected_sequence.as_bytes(), record.ref_qual()).unwrap();
        Ok(())
    }
}
fn extract_id_number(header: &str) -> Option<usize> {
    // Split the header by '-' or '.' and get the last part
    let id_part = header.split(&['-', '.'][..]).last()?;

    // Parse the ID part into a number
    id_part.parse::<usize>().ok()
}

#[pymodule]
fn fastq_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_parse_fastq, m)?)?;
    //m.add_function(wrap_pyfunction!(parse_fastq_file_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_fastq_foreach, m)?)?;
    m.add_function(wrap_pyfunction!(count_reads, m)?)?;
    m.add_function(wrap_pyfunction!(extract_kmers, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file_v2, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file_v3, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file_v4, m)?)?;
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
fn skip(reads_to_skip:u64, seqio_reader: &mut FastqReader<File>)->Result<(), &str> 
{
    let mut skipped_reads:u64 = 0;
    while let Some(_) = seqio_reader.next(){
        if skipped_reads == reads_to_skip{
            break;
        }
        skipped_reads += 1;
    }
    Ok(())
}
fn give_bytes_offset(seqio_reader: &FastqReader<File>)->u64{
    let position = seqio_reader.position();
    position.byte()
}
#[pyfunction]
fn count_reads(file_path:String)->u64{
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let parser = fastq::Parser::new(reader);
    let mut read_counts = 0;
    parser.each(|_|{
        read_counts += 1;
        return true;
    }).expect("Invalid fastq file");
    read_counts

}
//start offset signifies 
#[pyfunction]
fn parse_fastq_foreach(file_path:String, start_offset:u64, batch_size:u64) -> PyResult<Vec<String>>{
    
    let mut seqio_reader = FastqReader::from_path(file_path.clone()).unwrap();
    skip(start_offset, &mut seqio_reader).unwrap();
    
    //this is a different reader from above
    let file = File::open(file_path).unwrap();
    let mut reader = BufReader::new(file);
    
    reader.seek(SeekFrom::Start(give_bytes_offset(&seqio_reader))).unwrap();
    let parser = fastq::Parser::new(reader);

    let mut result = vec![];
    let mut count = 0;
    parser.each(|record|{
        if count  >= batch_size {
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
// #[pyfunction]
// fn parallel_write_fastq(dst_filename: String, src_filename: String, matrix: &Bound<'_, PyArray2<u8>>) -> PyResult<()>{
//     let mut writer = BufWriter::new(File::create(dst_filename).unwrap());
//     let reader = FastqReader::from_path(src_filename).unwrap();
    
//     let np_matrix = unsafe { matrix.as_array() };
//     let result: Result<Vec<String>, _> = np_matrix
//         .rows()
//         .into_iter()
//         .map(|row| {
//             let mut vector_row = row.to_vec();
//             byte_to_string(&mut vector_row)
//         })
//         .collect();
//     let unwrapped_result = result.unwrap();
//     let mutex_result = Arc::new(Mutex::new(unwrapped_result));
//     let worker_func = |record:seq_io::fastq::RefRecord, _| {        
//         let local_result = mutex_result.clone();
//         let id = record.id().unwrap().parse::<u64>().unwrap();
//         let corrected_sequence = local_result.lock().unwrap().get(id as usize).unwrap();
        
//         seq_io::fastq::write_to(writer, record.head(), corrected_sequence.as_bytes(), record.qual()).unwrap()
//     };
//     parallel_fastq(reader, 32, 12, worker_func, |_, _| {


//     });
//     Ok(())
// }
#[pyfunction]
fn parallel_parse_fastq(file_path:String) -> PyResult<Vec<String>>{
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let parser = fastq::Parser::new(reader);
    //this might be slow wrapping in arc mutex 
    let global_results = Arc::new(Mutex::new(vec![]));


    let collected_results:Result<Vec<()>, _> = parser.parallel_each(16,
        {
            //the worker closure that will be executed on each worker threads
            let results = Arc::clone(&global_results);
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
    match FastqReader::from_path(file_path) {
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
fn write_fastq_file_v4(dst_filename:String, src_filename: String, matrix:&Bound<'_, PyArray2<u8>>)-> PyResult<()>{
    let np_matrix = unsafe { matrix.as_array() };
    let result: Result<Vec<String>, _> = np_matrix
        .rows()
        .into_iter()
        .map(|row| {
            let mut vector_row = row.to_vec();
            byte_to_string(&mut vector_row)
        })
        .collect();
    let unwrapped_result = result.unwrap();
    let seqio_reader = FastqReader::from_path(src_filename).unwrap();
    let file = File::create(dst_filename).unwrap();
    let buffered_writer: Box<dyn Write + Send> = Box::new(BufWriter::new(file));
    let writer = Arc::new(Mutex::new(buffered_writer));
    let processor = WriteCalculation{
        corrected_2d_reads:unwrapped_result,
        writer
    };
    seqio_reader.process_parallel(processor, 24).unwrap();
    Ok(())
}
#[pyfunction]
fn write_fastq_file_v3(dst_filename:String, src_filename: String, matrix:&Bound<'_, PyArray2<u8>>)-> PyResult<()>{
    let np_matrix = unsafe { matrix.as_array() };
    let result: Result<Vec<String>, _> = np_matrix
        .rows()
        .into_iter()
        .map(|row| {
            let mut vector_row = row.to_vec();
            byte_to_string(&mut vector_row)
        })
        .collect();
    let unwrapped_result = result.unwrap();
    let seqio_reader = FastqReader::from_path(src_filename).unwrap();
    let file  = OpenOptions::new().create(true).append(true).open(dst_filename).expect("Error creating file instance");
    let writer = Arc::new(Mutex::new(BufWriter::new(file)));
    let work = {
        let writer = writer.clone();
        // Write the record to the file
        move |records:&mut RecordSet| {
            let mut writer = writer.lock().unwrap();
            for record in records.into_iter(){
                let id_number:usize = extract_id_number(record.id().unwrap()).unwrap();
                let corrected_sequence = unwrapped_result.get(id_number - 1).unwrap();
                seq_io::fastq::write_to(&mut *writer, record.head(), corrected_sequence.as_bytes(), record.qual()).unwrap();
            }
        }
    };
    read_parallel(seqio_reader, 24, 10, work, |record_set|{
        while let Some(_) = record_set.next() {   
        }
    });

    // parallel_fastq(seqio_reader, 4, 2, work, |_, _| {
    //     Some(())
    // }).unwrap();
    Ok(())
}
#[pyfunction]
fn write_fastq_file_v2(
    dst_filename: String,
    src_filename: String,
    matrix: &Bound<'_, PyArray2<u8>>,
)->PyResult<()>{
    let np_matrix = unsafe { matrix.as_array() };
    let result: Result<Vec<String>, _> = np_matrix
        .rows()
        .into_iter()
        .map(|row| {
            let mut vector_row = row.to_vec();
            byte_to_string(&mut vector_row)
        })
        .collect();
    let unwrapped_result = result.unwrap();
    let file = File::open(src_filename).unwrap();
    let mut writer = BufWriter::new(File::create(dst_filename).unwrap());
    let reader = BufReader::new(file);
    let parser = fastq::Parser::new(reader);
    let mut id = 0;
    
    //wtf is this ahahahaha 
    parser.each(|record| {
        let seq = unwrapped_result.get(id as usize).unwrap();
        let mut owned_record = record.to_owned_record();
        owned_record.seq = seq.as_bytes().to_vec();
        match owned_record.write(&mut writer){
            Ok(_) => {
                id += 1;
                return true;
            },
            _ => return false
        }
    }).unwrap();
    Ok(())
}
#[pyfunction]
fn write_fastq_file(
    dst_filename: String,
    src_filename: String,
    matrix: &Bound<'_, PyArray2<u8>>,
    offset: usize,
) -> PyResult<()> {
    let file  = OpenOptions::new().create(true).append(true).open(dst_filename).expect("Error creating file instance");
    let writer = BufWriter::new(file);
    let mut writer =  bio::io::fastq::Writer::new(writer);
       
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
    let records = reader.records().skip(offset);
    match result {
        Ok(rows_as_strings) => {
            for (record, row) in records.zip(rows_as_strings.iter()) {
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
