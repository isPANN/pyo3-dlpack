//! DLPack FFI types following the DLPack specification.
//!
//! These types match the C definitions from the DLPack header:
//! <https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h>

use std::ffi::c_void;

/// Device type codes as defined by DLPack.
///
/// These correspond to `DLDeviceType` in the DLPack specification.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DLDeviceType {
    /// CPU device
    Cpu = 1,
    /// CUDA GPU device
    Cuda = 2,
    /// Pinned CUDA CPU memory (allocated with cudaMallocHost)
    CudaHost = 3,
    /// OpenCL device
    OpenCL = 4,
    /// Vulkan device
    Vulkan = 7,
    /// Metal device (Apple)
    Metal = 8,
    /// VPI device
    Vpi = 9,
    /// ROCm device (AMD)
    Rocm = 10,
    /// ROCm host pinned memory
    RocmHost = 11,
    /// External DMA buffer
    ExtDev = 12,
    /// CUDA managed/unified memory
    CudaManaged = 13,
    /// Intel OneAPI device
    OneApi = 14,
    /// WebGPU device
    WebGpu = 15,
    /// Hexagon DSP device
    Hexagon = 16,
    /// MAIA accelerator
    Maia = 17,
}

impl DLDeviceType {
    /// Convert from raw u32 value.
    ///
    /// Returns `None` for unknown device types.
    pub fn from_raw(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::Cpu),
            2 => Some(Self::Cuda),
            3 => Some(Self::CudaHost),
            4 => Some(Self::OpenCL),
            7 => Some(Self::Vulkan),
            8 => Some(Self::Metal),
            9 => Some(Self::Vpi),
            10 => Some(Self::Rocm),
            11 => Some(Self::RocmHost),
            12 => Some(Self::ExtDev),
            13 => Some(Self::CudaManaged),
            14 => Some(Self::OneApi),
            15 => Some(Self::WebGpu),
            16 => Some(Self::Hexagon),
            17 => Some(Self::Maia),
            _ => None,
        }
    }
}

/// A device descriptor specifying where tensor data resides.
///
/// This corresponds to `DLDevice` in the DLPack specification.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DLDevice {
    /// The device type (CPU, CUDA, etc.)
    pub device_type: u32,
    /// The device ID (e.g., which GPU for multi-GPU systems)
    pub device_id: i32,
}

impl DLDevice {
    /// Create a new device descriptor.
    pub fn new(device_type: DLDeviceType, device_id: i32) -> Self {
        Self {
            device_type: device_type as u32,
            device_id,
        }
    }

    /// Get the device type as an enum.
    ///
    /// Returns `None` for unknown device types.
    pub fn device_type_enum(&self) -> Option<DLDeviceType> {
        DLDeviceType::from_raw(self.device_type)
    }

    /// Check if this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        self.device_type == DLDeviceType::Cuda as u32
    }

    /// Check if this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        self.device_type == DLDeviceType::Cpu as u32
    }

    /// Check if this is CUDA host (pinned) memory.
    pub fn is_cuda_host(&self) -> bool {
        self.device_type == DLDeviceType::CudaHost as u32
    }

    /// Check if this is a ROCm device.
    pub fn is_rocm(&self) -> bool {
        self.device_type == DLDeviceType::Rocm as u32
    }

    /// Check if this is a Metal device (Apple GPU).
    pub fn is_metal(&self) -> bool {
        self.device_type == DLDeviceType::Metal as u32
    }
}

/// Data type codes as defined by DLPack.
///
/// These correspond to `DLDataTypeCode` in the DLPack specification.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DLDataTypeCode {
    /// Signed integer
    Int = 0,
    /// Unsigned integer
    UInt = 1,
    /// IEEE floating point
    Float = 2,
    /// Opaque handle type (not for computation)
    OpaqueHandle = 3,
    /// Bfloat16 (Brain Floating Point)
    Bfloat = 4,
    /// Complex numbers
    Complex = 5,
    /// Boolean
    Bool = 6,
}

impl DLDataTypeCode {
    /// Convert from raw u8 value.
    ///
    /// Returns `None` for unknown type codes.
    pub fn from_raw(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Int),
            1 => Some(Self::UInt),
            2 => Some(Self::Float),
            3 => Some(Self::OpaqueHandle),
            4 => Some(Self::Bfloat),
            5 => Some(Self::Complex),
            6 => Some(Self::Bool),
            _ => None,
        }
    }
}

/// Data type descriptor specifying the element type of a tensor.
///
/// This corresponds to `DLDataType` in the DLPack specification.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DLDataType {
    /// Type code (signed int, unsigned int, float, etc.)
    pub code: u8,
    /// Number of bits per element (e.g., 32 for float32)
    pub bits: u8,
    /// Number of lanes for vectorized types (usually 1)
    pub lanes: u16,
}

impl DLDataType {
    /// Create a new data type descriptor.
    pub fn new(code: DLDataTypeCode, bits: u8, lanes: u16) -> Self {
        Self {
            code: code as u8,
            bits,
            lanes,
        }
    }

    /// Get the type code as an enum.
    ///
    /// Returns `None` for unknown type codes.
    pub fn code_enum(&self) -> Option<DLDataTypeCode> {
        DLDataTypeCode::from_raw(self.code)
    }

    /// Check if this is f16 (half precision float).
    pub fn is_f16(&self) -> bool {
        self.code == DLDataTypeCode::Float as u8 && self.bits == 16 && self.lanes == 1
    }

    /// Check if this is f32 (single precision float).
    pub fn is_f32(&self) -> bool {
        self.code == DLDataTypeCode::Float as u8 && self.bits == 32 && self.lanes == 1
    }

    /// Check if this is f64 (double precision float).
    pub fn is_f64(&self) -> bool {
        self.code == DLDataTypeCode::Float as u8 && self.bits == 64 && self.lanes == 1
    }

    /// Check if this is bf16 (bfloat16).
    pub fn is_bf16(&self) -> bool {
        self.code == DLDataTypeCode::Bfloat as u8 && self.bits == 16 && self.lanes == 1
    }

    /// Check if this is i8 (signed 8-bit integer).
    pub fn is_i8(&self) -> bool {
        self.code == DLDataTypeCode::Int as u8 && self.bits == 8 && self.lanes == 1
    }

    /// Check if this is i16 (signed 16-bit integer).
    pub fn is_i16(&self) -> bool {
        self.code == DLDataTypeCode::Int as u8 && self.bits == 16 && self.lanes == 1
    }

    /// Check if this is i32 (signed 32-bit integer).
    pub fn is_i32(&self) -> bool {
        self.code == DLDataTypeCode::Int as u8 && self.bits == 32 && self.lanes == 1
    }

    /// Check if this is i64 (signed 64-bit integer).
    pub fn is_i64(&self) -> bool {
        self.code == DLDataTypeCode::Int as u8 && self.bits == 64 && self.lanes == 1
    }

    /// Check if this is u8 (unsigned 8-bit integer).
    pub fn is_u8(&self) -> bool {
        self.code == DLDataTypeCode::UInt as u8 && self.bits == 8 && self.lanes == 1
    }

    /// Check if this is u16 (unsigned 16-bit integer).
    pub fn is_u16(&self) -> bool {
        self.code == DLDataTypeCode::UInt as u8 && self.bits == 16 && self.lanes == 1
    }

    /// Check if this is u32 (unsigned 32-bit integer).
    pub fn is_u32(&self) -> bool {
        self.code == DLDataTypeCode::UInt as u8 && self.bits == 32 && self.lanes == 1
    }

    /// Check if this is u64 (unsigned 64-bit integer).
    pub fn is_u64(&self) -> bool {
        self.code == DLDataTypeCode::UInt as u8 && self.bits == 64 && self.lanes == 1
    }

    /// Check if this is bool.
    pub fn is_bool(&self) -> bool {
        self.code == DLDataTypeCode::Bool as u8 && self.bits == 8 && self.lanes == 1
    }

    /// Get the size of one element in bytes.
    pub fn itemsize(&self) -> usize {
        ((self.bits as usize) * (self.lanes as usize) + 7) / 8
    }
}

/// The core DLTensor structure describing a tensor's data and layout.
///
/// This corresponds to `DLTensor` in the DLPack specification.
#[repr(C)]
pub struct DLTensor {
    /// Pointer to the data buffer.
    /// For GPU tensors, this is a device pointer.
    pub data: *mut c_void,
    /// Device descriptor specifying where the data resides.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Data type descriptor.
    pub dtype: DLDataType,
    /// Shape array (length = ndim).
    /// Points to an array of dimension sizes.
    pub shape: *mut i64,
    /// Stride array in number of elements (length = ndim).
    /// Can be null for compact row-major tensors.
    pub strides: *mut i64,
    /// Byte offset from the data pointer to the first element.
    pub byte_offset: u64,
}

/// Deleter function signature for DLManagedTensor.
///
/// Called when the consumer is done with the tensor to free resources.
pub type DLManagedTensorDeleter = unsafe extern "C" fn(*mut DLManagedTensor);

/// A managed tensor with ownership semantics.
///
/// This corresponds to `DLManagedTensor` in the DLPack specification.
/// It wraps a `DLTensor` and provides a deleter for cleanup.
#[repr(C)]
pub struct DLManagedTensor {
    /// The underlying tensor descriptor.
    pub dl_tensor: DLTensor,
    /// Opaque manager context for the producer's use.
    /// Typically used to store data needed by the deleter.
    pub manager_ctx: *mut c_void,
    /// Deleter function called when the consumer is done.
    /// Can be null if no cleanup is needed.
    pub deleter: Option<DLManagedTensorDeleter>,
}

// ============================================================================
// Convenience constructors
// ============================================================================

/// Create a DLDevice for CUDA with the specified device ID.
pub fn cuda_device(device_id: i32) -> DLDevice {
    DLDevice::new(DLDeviceType::Cuda, device_id)
}

/// Create a DLDevice for CPU.
pub fn cpu_device() -> DLDevice {
    DLDevice::new(DLDeviceType::Cpu, 0)
}

/// Create a DLDevice for Metal (Apple GPU) with the specified device ID.
pub fn metal_device(device_id: i32) -> DLDevice {
    DLDevice::new(DLDeviceType::Metal, device_id)
}

/// Create a DLDataType for f32 (single precision float).
pub fn dtype_f32() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Float, 32, 1)
}

/// Create a DLDataType for f64 (double precision float).
pub fn dtype_f64() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Float, 64, 1)
}

/// Create a DLDataType for f16 (half precision float).
pub fn dtype_f16() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Float, 16, 1)
}

/// Create a DLDataType for bf16 (bfloat16).
pub fn dtype_bf16() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Bfloat, 16, 1)
}

/// Create a DLDataType for i8 (signed 8-bit integer).
pub fn dtype_i8() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Int, 8, 1)
}

/// Create a DLDataType for i16 (signed 16-bit integer).
pub fn dtype_i16() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Int, 16, 1)
}

/// Create a DLDataType for i32 (signed 32-bit integer).
pub fn dtype_i32() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Int, 32, 1)
}

/// Create a DLDataType for i64 (signed 64-bit integer).
pub fn dtype_i64() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Int, 64, 1)
}

/// Create a DLDataType for u8 (unsigned 8-bit integer).
pub fn dtype_u8() -> DLDataType {
    DLDataType::new(DLDataTypeCode::UInt, 8, 1)
}

/// Create a DLDataType for u16 (unsigned 16-bit integer).
pub fn dtype_u16() -> DLDataType {
    DLDataType::new(DLDataTypeCode::UInt, 16, 1)
}

/// Create a DLDataType for u32 (unsigned 32-bit integer).
pub fn dtype_u32() -> DLDataType {
    DLDataType::new(DLDataTypeCode::UInt, 32, 1)
}

/// Create a DLDataType for u64 (unsigned 64-bit integer).
pub fn dtype_u64() -> DLDataType {
    DLDataType::new(DLDataTypeCode::UInt, 64, 1)
}

/// Create a DLDataType for bool.
pub fn dtype_bool() -> DLDataType {
    DLDataType::new(DLDataTypeCode::Bool, 8, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // DLDeviceType tests
    // ========================================================================

    #[test]
    fn test_device_type_from_raw_all_variants() {
        assert_eq!(DLDeviceType::from_raw(1), Some(DLDeviceType::Cpu));
        assert_eq!(DLDeviceType::from_raw(2), Some(DLDeviceType::Cuda));
        assert_eq!(DLDeviceType::from_raw(3), Some(DLDeviceType::CudaHost));
        assert_eq!(DLDeviceType::from_raw(4), Some(DLDeviceType::OpenCL));
        assert_eq!(DLDeviceType::from_raw(7), Some(DLDeviceType::Vulkan));
        assert_eq!(DLDeviceType::from_raw(8), Some(DLDeviceType::Metal));
        assert_eq!(DLDeviceType::from_raw(9), Some(DLDeviceType::Vpi));
        assert_eq!(DLDeviceType::from_raw(10), Some(DLDeviceType::Rocm));
        assert_eq!(DLDeviceType::from_raw(11), Some(DLDeviceType::RocmHost));
        assert_eq!(DLDeviceType::from_raw(12), Some(DLDeviceType::ExtDev));
        assert_eq!(DLDeviceType::from_raw(13), Some(DLDeviceType::CudaManaged));
        assert_eq!(DLDeviceType::from_raw(14), Some(DLDeviceType::OneApi));
        assert_eq!(DLDeviceType::from_raw(15), Some(DLDeviceType::WebGpu));
        assert_eq!(DLDeviceType::from_raw(16), Some(DLDeviceType::Hexagon));
        assert_eq!(DLDeviceType::from_raw(17), Some(DLDeviceType::Maia));
    }

    #[test]
    fn test_device_type_from_raw_unknown() {
        assert_eq!(DLDeviceType::from_raw(0), None);
        assert_eq!(DLDeviceType::from_raw(5), None);
        assert_eq!(DLDeviceType::from_raw(6), None);
        assert_eq!(DLDeviceType::from_raw(18), None);
        assert_eq!(DLDeviceType::from_raw(100), None);
        assert_eq!(DLDeviceType::from_raw(u32::MAX), None);
    }

    #[test]
    fn test_device_type_debug() {
        assert_eq!(format!("{:?}", DLDeviceType::Cpu), "Cpu");
        assert_eq!(format!("{:?}", DLDeviceType::Cuda), "Cuda");
    }

    #[test]
    fn test_device_type_clone_copy() {
        let dt = DLDeviceType::Cuda;
        let dt2 = dt;
        let dt3 = dt;
        assert_eq!(dt, dt2);
        assert_eq!(dt, dt3);
    }

    #[test]
    fn test_device_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DLDeviceType::Cpu);
        set.insert(DLDeviceType::Cuda);
        set.insert(DLDeviceType::Cpu);
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // DLDevice tests
    // ========================================================================

    #[test]
    fn test_device_new() {
        let dev = DLDevice::new(DLDeviceType::Cuda, 3);
        assert_eq!(dev.device_type, 2);
        assert_eq!(dev.device_id, 3);
    }

    #[test]
    fn test_device_type_enum() {
        let dev = DLDevice::new(DLDeviceType::Rocm, 1);
        assert_eq!(dev.device_type_enum(), Some(DLDeviceType::Rocm));

        let unknown = DLDevice { device_type: 99, device_id: 0 };
        assert_eq!(unknown.device_type_enum(), None);
    }

    #[test]
    fn test_device_is_cuda() {
        assert!(cuda_device(0).is_cuda());
        assert!(!cpu_device().is_cuda());
        assert!(!DLDevice::new(DLDeviceType::CudaHost, 0).is_cuda());
    }

    #[test]
    fn test_device_is_cpu() {
        assert!(cpu_device().is_cpu());
        assert!(!cuda_device(0).is_cpu());
    }

    #[test]
    fn test_device_is_cuda_host() {
        assert!(DLDevice::new(DLDeviceType::CudaHost, 0).is_cuda_host());
        assert!(!cpu_device().is_cuda_host());
        assert!(!cuda_device(0).is_cuda_host());
    }

    #[test]
    fn test_device_is_rocm() {
        assert!(DLDevice::new(DLDeviceType::Rocm, 0).is_rocm());
        assert!(!cpu_device().is_rocm());
        assert!(!cuda_device(0).is_rocm());
    }

    #[test]
    fn test_device_is_metal() {
        assert!(DLDevice::new(DLDeviceType::Metal, 0).is_metal());
        assert!(metal_device(0).is_metal());
        assert!(!cpu_device().is_metal());
        assert!(!cuda_device(0).is_metal());
    }

    #[test]
    fn test_device_debug() {
        let dev = cuda_device(2);
        let debug = format!("{:?}", dev);
        assert!(debug.contains("device_type"));
        assert!(debug.contains("device_id"));
    }

    #[test]
    fn test_device_clone_copy() {
        let dev = cuda_device(1);
        let dev2 = dev;
        let dev3 = dev;
        assert_eq!(dev, dev2);
        assert_eq!(dev, dev3);
    }

    #[test]
    fn test_device_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(cpu_device());
        set.insert(cuda_device(0));
        set.insert(cuda_device(1));
        set.insert(cpu_device());
        assert_eq!(set.len(), 3);
    }

    // ========================================================================
    // DLDataTypeCode tests
    // ========================================================================

    #[test]
    fn test_dtype_code_from_raw_all_variants() {
        assert_eq!(DLDataTypeCode::from_raw(0), Some(DLDataTypeCode::Int));
        assert_eq!(DLDataTypeCode::from_raw(1), Some(DLDataTypeCode::UInt));
        assert_eq!(DLDataTypeCode::from_raw(2), Some(DLDataTypeCode::Float));
        assert_eq!(DLDataTypeCode::from_raw(3), Some(DLDataTypeCode::OpaqueHandle));
        assert_eq!(DLDataTypeCode::from_raw(4), Some(DLDataTypeCode::Bfloat));
        assert_eq!(DLDataTypeCode::from_raw(5), Some(DLDataTypeCode::Complex));
        assert_eq!(DLDataTypeCode::from_raw(6), Some(DLDataTypeCode::Bool));
    }

    #[test]
    fn test_dtype_code_from_raw_unknown() {
        assert_eq!(DLDataTypeCode::from_raw(7), None);
        assert_eq!(DLDataTypeCode::from_raw(100), None);
        assert_eq!(DLDataTypeCode::from_raw(u8::MAX), None);
    }

    #[test]
    fn test_dtype_code_debug() {
        assert_eq!(format!("{:?}", DLDataTypeCode::Float), "Float");
        assert_eq!(format!("{:?}", DLDataTypeCode::Int), "Int");
    }

    #[test]
    fn test_dtype_code_clone_copy() {
        let code = DLDataTypeCode::Float;
        let code2 = code;
        let code3 = code;
        assert_eq!(code, code2);
        assert_eq!(code, code3);
    }

    #[test]
    fn test_dtype_code_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DLDataTypeCode::Float);
        set.insert(DLDataTypeCode::Int);
        set.insert(DLDataTypeCode::Float);
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // DLDataType tests
    // ========================================================================

    #[test]
    fn test_dtype_new() {
        let dt = DLDataType::new(DLDataTypeCode::Float, 32, 1);
        assert_eq!(dt.code, 2);
        assert_eq!(dt.bits, 32);
        assert_eq!(dt.lanes, 1);
    }

    #[test]
    fn test_dtype_code_enum() {
        let dt = dtype_f32();
        assert_eq!(dt.code_enum(), Some(DLDataTypeCode::Float));

        let unknown = DLDataType { code: 99, bits: 32, lanes: 1 };
        assert_eq!(unknown.code_enum(), None);
    }

    #[test]
    fn test_dtype_is_f16() {
        assert!(dtype_f16().is_f16());
        assert!(!dtype_f32().is_f16());
        assert!(!dtype_bf16().is_f16());
        // Wrong lanes
        let wrong = DLDataType::new(DLDataTypeCode::Float, 16, 2);
        assert!(!wrong.is_f16());
    }

    #[test]
    fn test_dtype_is_f32() {
        assert!(dtype_f32().is_f32());
        assert!(!dtype_f64().is_f32());
        assert!(!dtype_f16().is_f32());
    }

    #[test]
    fn test_dtype_is_f64() {
        assert!(dtype_f64().is_f64());
        assert!(!dtype_f32().is_f64());
    }

    #[test]
    fn test_dtype_is_bf16() {
        assert!(dtype_bf16().is_bf16());
        assert!(!dtype_f16().is_bf16());
        assert!(!dtype_f32().is_bf16());
    }

    #[test]
    fn test_dtype_is_i8() {
        assert!(dtype_i8().is_i8());
        assert!(!dtype_i16().is_i8());
        assert!(!dtype_u8().is_i8());
    }

    #[test]
    fn test_dtype_is_i16() {
        assert!(dtype_i16().is_i16());
        assert!(!dtype_i8().is_i16());
        assert!(!dtype_i32().is_i16());
    }

    #[test]
    fn test_dtype_is_i32() {
        assert!(dtype_i32().is_i32());
        assert!(!dtype_i64().is_i32());
        assert!(!dtype_u32().is_i32());
    }

    #[test]
    fn test_dtype_is_i64() {
        assert!(dtype_i64().is_i64());
        assert!(!dtype_i32().is_i64());
    }

    #[test]
    fn test_dtype_is_u8() {
        assert!(dtype_u8().is_u8());
        assert!(!dtype_i8().is_u8());
        assert!(!dtype_u16().is_u8());
    }

    #[test]
    fn test_dtype_is_u16() {
        assert!(dtype_u16().is_u16());
        assert!(!dtype_u8().is_u16());
    }

    #[test]
    fn test_dtype_is_u32() {
        assert!(dtype_u32().is_u32());
        assert!(!dtype_i32().is_u32());
    }

    #[test]
    fn test_dtype_is_u64() {
        assert!(dtype_u64().is_u64());
        assert!(!dtype_u32().is_u64());
    }

    #[test]
    fn test_dtype_is_bool() {
        assert!(dtype_bool().is_bool());
        assert!(!dtype_u8().is_bool());
        assert!(!dtype_i8().is_bool());
    }

    #[test]
    fn test_dtype_itemsize() {
        assert_eq!(dtype_f16().itemsize(), 2);
        assert_eq!(dtype_f32().itemsize(), 4);
        assert_eq!(dtype_f64().itemsize(), 8);
        assert_eq!(dtype_bf16().itemsize(), 2);
        assert_eq!(dtype_i8().itemsize(), 1);
        assert_eq!(dtype_i16().itemsize(), 2);
        assert_eq!(dtype_i32().itemsize(), 4);
        assert_eq!(dtype_i64().itemsize(), 8);
        assert_eq!(dtype_u8().itemsize(), 1);
        assert_eq!(dtype_u16().itemsize(), 2);
        assert_eq!(dtype_u32().itemsize(), 4);
        assert_eq!(dtype_u64().itemsize(), 8);
        assert_eq!(dtype_bool().itemsize(), 1);
    }

    #[test]
    fn test_dtype_itemsize_vectorized() {
        // Vectorized type with 4 lanes of f32
        let vec_f32 = DLDataType::new(DLDataTypeCode::Float, 32, 4);
        assert_eq!(vec_f32.itemsize(), 16); // 4 * 4 bytes

        // 8 lanes of i16
        let vec_i16 = DLDataType::new(DLDataTypeCode::Int, 16, 8);
        assert_eq!(vec_i16.itemsize(), 16); // 8 * 2 bytes
    }

    #[test]
    fn test_dtype_itemsize_rounding() {
        // Test rounding up for non-byte-aligned types
        let one_bit = DLDataType { code: 0, bits: 1, lanes: 1 };
        assert_eq!(one_bit.itemsize(), 1);

        let seven_bits = DLDataType { code: 0, bits: 7, lanes: 1 };
        assert_eq!(seven_bits.itemsize(), 1);

        let nine_bits = DLDataType { code: 0, bits: 9, lanes: 1 };
        assert_eq!(nine_bits.itemsize(), 2);
    }

    #[test]
    fn test_dtype_debug() {
        let dt = dtype_f32();
        let debug = format!("{:?}", dt);
        assert!(debug.contains("code"));
        assert!(debug.contains("bits"));
        assert!(debug.contains("lanes"));
    }

    #[test]
    fn test_dtype_clone_copy() {
        let dt = dtype_f32();
        let dt2 = dt;
        let dt3 = dt;
        assert_eq!(dt, dt2);
        assert_eq!(dt, dt3);
    }

    #[test]
    fn test_dtype_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(dtype_f32());
        set.insert(dtype_f64());
        set.insert(dtype_f32());
        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // Convenience constructor tests
    // ========================================================================

    #[test]
    fn test_cuda_device() {
        let dev = cuda_device(0);
        assert!(dev.is_cuda());
        assert_eq!(dev.device_id, 0);

        let dev1 = cuda_device(1);
        assert!(dev1.is_cuda());
        assert_eq!(dev1.device_id, 1);
    }

    #[test]
    fn test_cpu_device() {
        let dev = cpu_device();
        assert!(dev.is_cpu());
        assert_eq!(dev.device_id, 0);
    }

    #[test]
    fn test_metal_device() {
        let dev = metal_device(0);
        assert!(dev.is_metal());
        assert_eq!(dev.device_id, 0);

        let dev1 = metal_device(1);
        assert!(dev1.is_metal());
        assert_eq!(dev1.device_id, 1);
    }

    #[test]
    fn test_all_dtype_constructors() {
        // Float types
        assert!(dtype_f16().is_f16());
        assert!(dtype_f32().is_f32());
        assert!(dtype_f64().is_f64());
        assert!(dtype_bf16().is_bf16());

        // Signed integer types
        assert!(dtype_i8().is_i8());
        assert!(dtype_i16().is_i16());
        assert!(dtype_i32().is_i32());
        assert!(dtype_i64().is_i64());

        // Unsigned integer types
        assert!(dtype_u8().is_u8());
        assert!(dtype_u16().is_u16());
        assert!(dtype_u32().is_u32());
        assert!(dtype_u64().is_u64());

        // Boolean
        assert!(dtype_bool().is_bool());
    }

    // ========================================================================
    // DLTensor and DLManagedTensor struct layout tests
    // ========================================================================

    #[test]
    fn test_dl_tensor_size() {
        // DLTensor should be well-defined for FFI
        // This test ensures the struct has a reasonable size
        let size = std::mem::size_of::<DLTensor>();
        assert!(size > 0);
        // On 64-bit systems: data(8) + device(8) + ndim(4) + dtype(4) + shape(8) + strides(8) + byte_offset(8)
        // = 48 bytes (with padding)
    }

    #[test]
    fn test_dl_managed_tensor_size() {
        let size = std::mem::size_of::<DLManagedTensor>();
        assert!(size > 0);
        // DLManagedTensor = DLTensor + manager_ctx(8) + deleter(8 or 16 for Option<fn>)
    }

    #[test]
    fn test_dl_device_repr_c() {
        // Verify the struct has expected alignment for FFI
        assert_eq!(std::mem::align_of::<DLDevice>(), 4);
        assert_eq!(std::mem::size_of::<DLDevice>(), 8);
    }

    #[test]
    fn test_dl_data_type_repr_c() {
        // Verify the struct has expected size for FFI
        assert_eq!(std::mem::size_of::<DLDataType>(), 4);
    }
}
