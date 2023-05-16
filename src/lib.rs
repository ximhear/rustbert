use libc::{c_char, c_int};
use std::ffi::CStr;
use tokenizers::{Tokenizer, EncodeInput, InputSequence};

#[no_mangle]
pub extern "C" fn process_string(input: *const c_char) -> *mut ProcessedResult {
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
    let masks = encoding.get_attention_mask();
    let result_vec: Vec<c_int> = ids.iter().map(|&id| id as c_int).collect();
    let mask_vec: Vec<c_int> = masks.iter().map(|&mask| mask as c_int).collect();
    println!("{:?}", ids);
    println!("{:?}", masks);

    // Allocate memory for the ProcessedResult struct in C++ heap
    let result = Box::new(ProcessedResult {
        result_vec: result_vec.as_ptr(),
        result_len: result_vec.len(),
        mask_vec: mask_vec.as_ptr(),
        mask_len: mask_vec.len(),
        result_box: Some(result_vec),
        mask_box: Some(mask_vec),
    });

    Box::into_raw(result)
}

#[no_mangle]
pub extern "C" fn free_processed_result(ptr: *mut ProcessedResult) {
    if !ptr.is_null() {
        let result = unsafe { Box::from_raw(ptr) };
        drop(result.result_box);
        drop(result.mask_box);
    }
}

#[repr(C)]
pub struct ProcessedResult {
    result_vec: *const c_int,
    result_len: usize,
    mask_vec: *const c_int,
    mask_len: usize,
    result_box: Option<Vec<c_int>>,
    mask_box: Option<Vec<c_int>>,
}