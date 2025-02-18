use bio::io::fastq::{self, Record};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::{thread, time::*};
use rayon::prelude::*;
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
                    let string_record = byte_to_string(record.seq());
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
    fn write_fastq_file(file_name:String, mut matrix:Vec<Vec<u8>>) -> PyResult<()>{
    let mut writer = fastq::Writer::to_file(file_name).unwrap();

    matrix = remove_zeros(matrix);
    let string_matrix = translate_numeric(matrix).unwrap();
    println!("{:?}",string_matrix);
    Ok(())
}

fn translate_numeric(matrix:Vec<Vec<u8>>) -> Result<Vec<String>,&'static str >{
    println!("translating matrix");
    let string_matrix:Result<Vec<String>, _> = matrix
        .into_par_iter()
        .map(|mut row|{

        for i in 0..row.len(){
            match row[i]{
                1 => row[i] = 65,
                2 => row[i] = 67,
                3 => row[i] = 71,
                4 => row[i] = 84,
                _ => return Err("Invalid value in matrix")
            }
        }
        Ok(byte_to_string(&row))

        }).collect();
    string_matrix
}

fn remove_zeros(matrix: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut new_matrix = vec![];
    for row in matrix {
        let mut new_row = vec![];
        for element in row {
            if element != 0 {
                new_row.push(element);
            }
        }
        new_matrix.push(new_row);
    }
    new_matrix
}
//this function unwraps for now. Add better error handling
fn byte_to_string(byte_array: &[u8]) -> String {
    std::str::from_utf8(byte_array).unwrap().to_string()
}
 
