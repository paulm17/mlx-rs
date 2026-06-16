use std::ffi::c_int;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxString {
    pub ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MlxDevice {
    pub ctx: *mut std::ffi::c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxDeviceType {
    Cpu = 0,
    Gpu = 1,
}

impl TryFrom<c_int> for MlxDeviceType {
    type Error = anyhow::Error;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(MlxDeviceType::Cpu),
            1 => Ok(MlxDeviceType::Gpu),
            other => Err(anyhow::anyhow!("unknown device type: {other}")),
        }
    }
}

pub type MlxVersionFn = unsafe extern "C" fn(*mut MlxString) -> c_int;
pub type MlxStringDataFn = unsafe extern "C" fn(MlxString) -> *const std::ffi::c_char;
pub type MlxStringFreeFn = unsafe extern "C" fn(MlxString) -> c_int;
pub type MlxDeviceIsAvailableFn = unsafe extern "C" fn(*mut bool, MlxDevice) -> c_int;
pub type MlxGetDefaultDeviceFn = unsafe extern "C" fn(*mut MlxDevice) -> c_int;
pub type MlxDeviceGetTypeFn = unsafe extern "C" fn(*mut c_int, MlxDevice) -> c_int;
pub type MlxDeviceCountFn = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
pub type MlxDeviceFreeFn = unsafe extern "C" fn(MlxDevice) -> c_int;

pub struct MlxSymbols {
    pub mlx_version: MlxVersionFn,
    pub mlx_string_data: MlxStringDataFn,
    pub mlx_string_free: MlxStringFreeFn,
    pub mlx_device_is_available: MlxDeviceIsAvailableFn,
    pub mlx_get_default_device: MlxGetDefaultDeviceFn,
    pub mlx_device_get_type: MlxDeviceGetTypeFn,
    pub mlx_device_count: MlxDeviceCountFn,
    pub mlx_device_free: MlxDeviceFreeFn,
}

unsafe impl Send for MlxSymbols {}

impl MlxSymbols {
    pub fn load(lib: &libloading::Library) -> anyhow::Result<Self> {
        unsafe {
            let mlx_version: MlxVersionFn = *lib
                .get(b"mlx_version\0")
                .map_err(|e| anyhow::anyhow!("symbol mlx_version not found: {e}"))?;

            let mlx_string_data: MlxStringDataFn = *lib
                .get(b"mlx_string_data\0")
                .map_err(|e| anyhow::anyhow!("symbol mlx_string_data not found: {e}"))?;

            let mlx_string_free: MlxStringFreeFn = *lib
                .get(b"mlx_string_free\0")
                .map_err(|e| anyhow::anyhow!("symbol mlx_string_free not found: {e}"))?;

            let mlx_device_is_available: MlxDeviceIsAvailableFn = *lib
                .get(b"mlx_device_is_available\0")
                .map_err(|e| {
                    anyhow::anyhow!("symbol mlx_device_is_available not found: {e}")
                })?;

            let mlx_get_default_device: MlxGetDefaultDeviceFn = *lib
                .get(b"mlx_get_default_device\0")
                .map_err(|e| {
                    anyhow::anyhow!("symbol mlx_get_default_device not found: {e}")
                })?;

            let mlx_device_get_type: MlxDeviceGetTypeFn = *lib
                .get(b"mlx_device_get_type\0")
                .map_err(|e| anyhow::anyhow!("symbol mlx_device_get_type not found: {e}"))?;

            let mlx_device_count: MlxDeviceCountFn = *lib
                .get(b"mlx_device_count\0")
                .map_err(|e| anyhow::anyhow!("symbol mlx_device_count not found: {e}"))?;

            let mlx_device_free: MlxDeviceFreeFn = *lib
                .get(b"mlx_device_free\0")
                .map_err(|e| anyhow::anyhow!("symbol mlx_device_free not found: {e}"))?;

            Ok(Self {
                mlx_version,
                mlx_string_data,
                mlx_string_free,
                mlx_device_is_available,
                mlx_get_default_device,
                mlx_device_get_type,
                mlx_device_count,
                mlx_device_free,
            })
        }
    }
}
