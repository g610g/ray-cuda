use bio::io::fastq::{self, Record};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods};

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
            return Err(PyErr::new::<PyTypeError, _>("Source fastq file name cannot be located"));
        }
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
