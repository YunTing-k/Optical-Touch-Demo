import ctypes
import logging
import time

import numpy

sys_log = logging.getLogger('logger')
ftd_status = {
    0: 'FT_OK',
    1: 'FT_INVALID_HANDLE',
    2: 'FT_DEVICE_NOT_FOUND',
    3: 'FT_DEVICE_NOT_OPENED',
    4: 'FT_IO_ERROR',
    5: 'FT_INSUFFICIENT_RESOURCES',
    6: 'FT_INVALID_PARAMETER',
    7: 'FT_INVALID_BAUD_RATE',
    8: 'FT_DEVICE_NOT_OPENED_FOR_ERASE',
    9: 'FT_DEVICE_NOT_OPENED_FOR_WRITE',
    10: 'FT_FAILED_TO_WRITE_DEVICE',
    11: 'FT_EEPROM_READ_FAILED',
    12: 'FT_EEPROM_WRITE_FAILED',
    13: 'FT_EEPROM_ERASE_FAILED',
    14: 'FT_EEPROM_NOT_PRESENT',
    15: 'FT_EEPROM_NOT_PROGRAMMED',
    16: 'FT_INVALID_ARGS',
    17: 'FT_NOT_SUPPORTED',
    18: 'FT_OTHER_ERROR',
    19: 'FT_DEVICE_LIST_NOT_READY'
}


def create_ftd_dll():  # 加载ftd2xx64 DLL用于后续调用
    # 打开DLL
    dll = ctypes.cdll.LoadLibrary('./ftd2xx64.dll')
    return dll


def ftd_handle(dll, open_flag=1, serial_bytes=b'FT7UBM74'):  # 打开设备并且创建device handle
    serial_str = ctypes.c_char_p(serial_bytes)  # 序列号 char pointer类型
    device_open_flag = ctypes.c_ulong(open_flag)  # 1-通过序列号打开 2-通过描述打开 4-通过位置打开
    device_handle = ctypes.c_void_p()  # 设备的handle
    device_status = dll.FT_OpenEx(serial_str, device_open_flag, ctypes.byref(device_handle))
    if not device_status:
        sys_log.info('USB设备打开成功')
    else:
        sys_log.error('USB设备打开失败')
        sys_log.error(ftd_status[device_status])
    return device_handle


def ftd_close(dll, device_handle):  # 关闭ftd设备
    device_status = dll.FT_Close(device_handle)
    if not device_status:
        sys_log.info('USB设备关闭成功')
    else:
        sys_log.error('USB设备关闭失败')
        sys_log.error(ftd_status[device_status])


def ftd_purge(dll, device_handle, rx=True, tx=True):  # 清除ftd设备的缓存
    purge_mask = ctypes.c_ulong(rx * 1 + tx * 2)  # 清除缓存标志
    device_status = dll.FT_Purge(device_handle, purge_mask)  # 获取设备状态
    sys_log.debug('接收缓存清除:[%s], 发送缓存清除:[%s]' % (str(rx), str(tx)))
    if not device_status:
        sys_log.info('缓存清除成功')
    else:
        sys_log.error('缓存清除失败')
        sys_log.error(ftd_status[device_status])


def ftd_set_bit_mode(dll, device_handle, mask=0, mode=0x40):  # 设置Chip mode
    _mask_ = ctypes.c_ubyte(mask)
    _mode_ = ctypes.c_ubyte(mode)
    device_status = dll.FT_SetBitMode(device_handle, _mask_, _mode_)  # 获取设备状态
    sys_log.debug('mask:[%s], mode:[%s]' % (str(mask), hex(mode)))
    if not device_status:
        sys_log.info('模式设置成功')
    else:
        sys_log.error('模式设置失败')
        sys_log.error(ftd_status[device_status])


def ftd_set_usb_param(dll, device_handle, in_size=65536, out_size=65536):  # 设置传输大小
    _in_size_ = ctypes.c_ulong(in_size)
    _out_size_ = ctypes.c_ulong(out_size)
    device_status = dll.FT_SetUSBParameters(device_handle, _in_size_, _out_size_)  # 获取设备状态
    sys_log.debug('输入请求大小:%d bytes, 输出请求大小:%d bytes' % (in_size, out_size))
    if not device_status:
        sys_log.info('传输大小设置成功')
    else:
        sys_log.error('传输大小设置失败')
        sys_log.error(ftd_status[device_status])


def ftd_set_flow_ctr(dll, device_handle, flow_ctr=0x0100, u_xon=0, u_xoff=0):  # 设置流量控制
    _flow_ctr_ = ctypes.c_ushort(flow_ctr)
    _u_xon_ = ctypes.c_ubyte(u_xon)
    _u_xoff_ = ctypes.c_ubyte(u_xoff)
    device_status = dll.FT_SetFlowControl(device_handle, _flow_ctr_, _u_xon_, _u_xoff_)  # 获取设备状态
    sys_log.debug('流量控制:[%s], uxon:%d, uxoff:%d' % (hex(flow_ctr), u_xon, u_xoff))
    if not device_status:
        sys_log.info('流量控制设置成功')
    else:
        sys_log.error('流量控制设置失败')
        sys_log.error(ftd_status[device_status])


def ftd_set_timeout(dll, device_handle, read_timeout=1000, write_timeout=1000):  # 设置超时
    _read_timeout_ = ctypes.c_ulong(read_timeout)
    _write_timeout_ = ctypes.c_ulong(write_timeout)
    device_status = dll.FT_SetTimeouts(device_handle, _read_timeout_, _write_timeout_)  # 获取设备状态
    sys_log.debug('读取超时设置:%d ms, 写入超时设置:%d ms' % (read_timeout, write_timeout))
    if not device_status:
        sys_log.info('超时设置成功')
    else:
        sys_log.error('超时设置失败')
        sys_log.error(ftd_status[device_status])


def ftd_send_check(device_status, bytes_written, bytes_to_write):  # 检查发送情况
    if not device_status:
        # sys_log.debug('应该发送%d bytes, 实际发送%d bytes' % (bytes_to_write.value, bytes_written.value))
        if bytes_written.value == bytes_to_write.value:
            # sys_log.info('数据发送成功')
            pass
        else:
            sys_log.warning('数据发送异常')
    else:
        sys_log.error('数据发送失败')
        sys_log.error(ftd_status[device_status])


def ftd_read_check(device_status, bytes_received, bytes_to_read):  # 检查接收情况
    if not device_status:
        # sys_log.debug('应该接收%d bytes, 实际接收%d bytes' % (bytes_to_read.value, bytes_received.value))
        if bytes_received.value == bytes_to_read.value:
            # sys_log.info('数据接收成功')
            pass
        else:
            sys_log.warning('数据接收异常')
    else:
        sys_log.error('数据接收失败')
        sys_log.error(ftd_status[device_status])


def ftd_send_data(dll, device_handle, data):  # 发送指定的数据
    send_buffer = bytes.fromhex(data)  # 转为bytes 送入缓冲区
    bytes_to_write = ctypes.c_ulong(len(send_buffer))  # 要发送的数据
    bytes_written = ctypes.c_void_p()  # 已经发送的bytes
    device_status = dll.FT_Write(device_handle, send_buffer, bytes_to_write, ctypes.byref(bytes_written))  # 获取设备状态
    ftd_send_check(device_status, bytes_written, bytes_to_write)  # 检查发送情况


def ftd_reset(dll, device_handle):  # 发送指令重置设备
    send_buffer = b'\xFF\xAA\x01\x00\x0D\xEB'  # 重置命令 送入缓冲区
    bytes_to_write = ctypes.c_ulong(len(send_buffer))  # 要发送的数据
    bytes_written = ctypes.c_void_p()  # 已经发送的bytes
    device_status = dll.FT_Write(device_handle, send_buffer, bytes_to_write, ctypes.byref(bytes_written))  # 获取设备状态
    ftd_send_check(device_status, bytes_written, bytes_to_write)  # 检查发送情况
    if not device_status:
        sys_log.info('系统重置成功')
    else:
        sys_log.error('系统设置失败')
        sys_log.error(ftd_status[device_status])


def ftd_TFT_config(dll, device_handle, v_bias3=3, v_bias4=-5, v_on=10, v_off=-10):  # 配置读出阵列的
    command1 = 'FFAA00A10DEB'  # 指令头
    command2 = '80007FFF'  # GOA_DC2与GOA_DC1
    command2 = command2 + numpy.uint16((v_off + 20) * 65536 / 40).tobytes()[::-1].hex()  # v_off (VGL)
    command2 = command2 + 'A0004000'  # TFT_VBias与TFT_VCOM
    # command2 = command2 + 'A0006000'  # TFT_VBias与TFT_VCOM
    command2 = command2 + numpy.uint16((v_on + 20) * 65536 / 40).tobytes()[::-1].hex()  # v_on (GOA_DCH)
    command2 = command2 + '800040004000'  # GOA_DC3, MUX_H与MUX_L
    # command2 = command2 + '800020002000'  # GOA_DC3, MUX_H与MUX_L
    command2 = command2 + numpy.uint16((v_bias3 + 20) * 65536 / 40).tobytes()[::-1].hex()  # v_bias3 (Pixel_DC1)
    command2 = command2 + numpy.uint16((v_bias4 + 20) * 65536 / 40).tobytes()[::-1].hex()  # v_bias4 (Pixel_DC2)
    command2 = command2 + '8000400020002000E000'  # Pixel_DC3, VClk_H, VScan_L, VScan_H与BK1
    # command2 = command2 + '80002000200020008000'  # Pixel_DC3, VClk_H, VScan_L, VScan_H与BK1
    command2_byte = bytes.fromhex(command2)
    command2_uint = numpy.frombuffer(command2_byte[::-1], dtype=numpy.uint8)
    command2_sum = numpy.sum(command2_uint)
    check_sum = numpy.uint8(numpy.mod(command2_sum, 256))
    command = command1 + command2 + check_sum.tobytes()[::-1].hex()
    # command = 'FFAA00A10DEB80007FFF4000A0006000C000800020002000933360008000200020002000800032'  # 参考指令-1 3 -5 10 -10
    # command = 'FFAA00A10DEB80007FFF4000A0004000C000800040004000933360008000400020002000E000E4'  # 参考指令-2 3 -5 10 -10
    # command = 'FFAA00A10DEB80007FFF3333A0004000CCCD800040004000933360008000400020002000E000E3'  # 参考指令-2 3 -5 12 -12
    # command = 'FFAA00A10DEB80007FFF3333A0004000CCCD800040004000933353338000400020002000E00009'  # 参考指令-2 3 -5 12 -12
    sys_log.debug('Vbias3=%.2f V, Vbias4=%.2f V, Von=%.2f V, Vbias=%.2f V' % (v_bias3, v_bias4, v_on, v_off))
    send_buffer = bytes.fromhex(command)  # 转为bytes 送入缓冲区
    bytes_to_write = ctypes.c_ulong(len(send_buffer))  # 要发送的数据
    bytes_written = ctypes.c_void_p()  # 已经发送的bytes
    device_status = dll.FT_Write(device_handle, send_buffer, bytes_to_write, ctypes.byref(bytes_written))  # 获取设备状态
    ftd_send_check(device_status, bytes_written, bytes_to_write)  # 检查发送情况


def ftd_read_part(dll, part, device_handle):  # 读出阵列的某一部分， part=1/2/3/4
    command = 'FFAA00'  # 发送读出命令
    command = command + '0' + str(part - 1) + '0DEB'
    send_buffer = bytes.fromhex(command)  # 转为bytes 送入缓冲区
    bytes_to_write = ctypes.c_ulong(len(send_buffer))  # 要发送的数据
    bytes_written = ctypes.c_void_p()  # 已经发送的bytes
    device_status1 = dll.FT_Write(device_handle, send_buffer, bytes_to_write, ctypes.byref(bytes_written))  # 检查发送情况
    ftd_send_check(device_status1, bytes_written, bytes_to_write)

    time.sleep(0.001)
    receive_buffer = ctypes.create_string_buffer(65536)
    bytes_to_read = ctypes.c_ulong(65536)  # 要读取的数据
    bytes_received = ctypes.c_void_p()  # 已经读取的bytes
    device_status2 = dll.FT_Read(device_handle, receive_buffer, bytes_to_read, ctypes.byref(bytes_received))  # 获取设备状态
    ftd_read_check(device_status2, bytes_received, bytes_to_read)  # 检查接收情况
    return receive_buffer.raw


def ftd_read_part_while(dll, part, device_handle):  # 读出阵列的某一部分， part=1/2/3/4
    command = 'FFAA00'  # 发送读出命令
    command = command + '0' + str(part - 1) + '0DEB'
    send_buffer = bytes.fromhex(command)  # 转为bytes 送入缓冲区
    bytes_to_write = ctypes.c_ulong(len(send_buffer))  # 要发送的数据
    bytes_written = ctypes.c_void_p()  # 已经发送的bytes
    device_status1 = dll.FT_Write(device_handle, send_buffer, bytes_to_write, ctypes.byref(bytes_written))  # 检查发送情况
    ftd_send_check(device_status1, bytes_written, bytes_to_write)

    rx_bytes_count = ctypes.c_ulong()
    tx_bytes_count = ctypes.c_ulong()
    event = ctypes.c_ulong()
    receive_buffer = ctypes.create_string_buffer(65536)
    bytes_received = ctypes.c_ulong()  # 已经读取的bytes数
    count = 0  # 缓冲区为空的计数
    time.sleep(0.001)
    while True:
        _ = dll.FT_GetStatus(device_handle, ctypes.byref(rx_bytes_count), ctypes.byref(tx_bytes_count), ctypes.byref(event))
        device_status = dll.FT_Read(device_handle, receive_buffer, rx_bytes_count, ctypes.byref(bytes_received))  # 获取设备状态
        # sys_log.debug(ftd_status[device_status])
        if rx_bytes_count.value != 0:
            # sys_log.warning(rx_bytes_count.value)
            pass
        else:
            time.sleep(0.001)
            count += 1
            if count >= 10:
                # sys_log.error('No Device Connected')
                break
    return receive_buffer.raw


def ftd_get_img(dll, device_handle):  # 得到全部的图像
    raw1 = ftd_read_part(dll, 1, device_handle)
    raw2 = ftd_read_part(dll, 2, device_handle)
    raw3 = ftd_read_part(dll, 3, device_handle)
    raw4 = ftd_read_part(dll, 4, device_handle)
    # raw1 = ftd_read_part_while(dll, 1, device_handle)
    # raw2 = ftd_read_part_while(dll, 2, device_handle)
    # raw3 = ftd_read_part_while(dll, 3, device_handle)
    # raw4 = ftd_read_part_while(dll, 4, device_handle)
    byteDATA = raw1 + raw2 + raw3 + raw4
    # uint32data = numpy.frombuffer(byteDATA[::-1], dtype=numpy.uint32)  # 4字节一读 反转
    uint32data = numpy.frombuffer(byteDATA, dtype=numpy.uint32)  # 4字节一读
    uint32data.shape = (256, 256)  # 转为256×256的uint32 array
    uint32data = numpy.rot90(uint32data, -1)
    # sys_log.debug('mean=%.4e, std=%.4e, max=%.4e, min=%.4e' %
    #               (numpy.mean(uint32data), numpy.std(uint32data), numpy.max(uint32data), numpy.min(uint32data)))
    return uint32data
