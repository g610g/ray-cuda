use bio::io::fastq::{self, Record};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::{thread, time::*};
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};

#[pymodule]
fn fastq_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_fastq_file, m)?)?;
    m.add_function(wrap_pyfunction!(write_fastq_file, m)?)?;
    Ok(())
}

//creates python bindings that will be used for parsing fastq files
#[pyfunction]
fn parse_fastq_file(file_path: String) -> PyResult<Vec<String>> {
    match fastq::Reader::from_file(file_path) {
        Ok(reader) => {
            let mut read_vector = vec![];
            for result in reader.records() {
                if let Ok(record) = result {
                    let string_record = byte_to_string(record.seq())?;
                    read_vector.push(string_record);
                }
            }
            return Ok(read_vector);
        }
        Err(_) => {
            //code panics for now. Will add better error handling
            panic!("Something went wrong during finding the file")
        }
    }
}
#[pyfunction]
fn write_fastq_file(file_name:String, matrix:&Bound<'_, PyArray2<u8>>) -> PyResult<()>{
    let mut writer = fastq::Writer::to_file(file_name).unwrap();

    unsafe {
        let np_matrix = matrix.as_array();
        let result :Result<Vec<String>, _> = np_matrix
            .rows()
            .into_iter().map(|row|{
                let vector_row = row.to_vec();
                byte_to_string(&vector_row)
        }).collect();
    };

    Ok(())
}

//this function unwraps for now. Add better error handling
fn byte_to_string(byte_array: &[u8]) -> Result<String, PyErr>{
    match std::str::from_utf8(byte_array){
        Ok(string) => Ok(string.to_string()),
        Err(_) => Err(PyErr::new::<PyTypeError, _>("Invalid bytes array"))

    }
}
