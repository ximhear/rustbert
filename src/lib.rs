use libc::{c_char, c_int};
use std::ffi::CStr;
use tokenizers::{Tokenizer, EncodeInput, InputSequence};

#[no_mangle]
pub extern "C" fn process_string(input: *const c_char) -> *mut Vec<c_int> {
    let input_str = unsafe {
        assert!(!input.is_null());
        CStr::from_ptr(input).to_string_lossy().into_owned()
    };
    println!("{:?}", input_str);

    // Process the string and generate the vector of integers
    let tokenizer = match Tokenizer::from_pretrained("bert-base-multilingual-cased", None) {
        Ok(tokenizer) => tokenizer,
        Err(_) => return std::ptr::null_mut(),
    };

    let encoding = match tokenizer.encode(EncodeInput::Single(InputSequence::from(input_str)), false) {
        Ok(encoding) => encoding,
        Err(_) => return std::ptr::null_mut(),
    };

    let ids = encoding.get_ids();
    println!("{:?}", ids);
    println!("{:?}", encoding.get_attention_mask());
    let result_vec: Vec<c_int> = ids.iter().map(|&id| id as c_int).collect();
    println!("{:?}", result_vec);

    // Allocate memory for the vector in C++ heap
    let result_ptr = Box::into_raw(Box::new(result_vec));
    println!("{:?}", result_ptr);

    result_ptr
}

#[no_mangle]
pub extern "C" fn get_result_size(result: *const Vec<c_int>) -> usize {
    let result_vec = unsafe { &*result };
    result_vec.len()
}

#[no_mangle]
pub extern "C" fn get_result_data(result: *const Vec<c_int>) -> *const c_int {
    let result_vec = unsafe { &*result };
    result_vec.as_ptr()
}

#[no_mangle]
pub extern "C" fn free_result(result: *mut Vec<c_int>) {
    if !result.is_null() {
        unsafe {
            Box::from_raw(result);
        }
    }
}