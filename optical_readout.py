from socket import *
import logging
import numpy

sys_log = logging.getLogger('logger')


def create_tcp_client():
    """创建TCP Client并且连接读出系统"""
    tcp_client_socket = socket(AF_INET, SOCK_STREAM)
    tcp_client_socket.setsockopt(SOL_SOCKET, SO_KEEPALIVE, 1)  # 在客户端开启心跳维护
    tcp_client_socket.setsockopt(SOL_SOCKET, SO_RCVBUF, 163880)  # 更改缓冲区大小
    server_ip = '192.168.137.219'
    server_port = 3192
    sys_log.debug('开始连接TCP')
    tcp_client_socket.connect((server_ip, server_port))
    sys_log.info('TCP连接成功')
    optical_readout_config(client=tcp_client_socket)  # 默认配置
    return tcp_client_socket


def start_stream(client):
    """开始获取数据流"""
    sys_log.warning('开始TCP数据流')
    client.send(bytes.fromhex('FF01010D0A'))


def stop_stream(client):
    """停止获取数据流并清空缓冲区"""
    sys_log.warning('停止TCP数据流')
    client.send(bytes.fromhex('FF01000D0A'))


def frame_valid(byte_data):
    """检测frame是否是正确的帧"""
    _str_ = byte_data[0:2].hex() + byte_data[16386:16388].hex()
    if _str_ == '12344321':
        sys_log.debug('TCP帧有效')
        pass
    else:
        sys_log.warning('TCP帧错误')


def optical_data_get(client):
    """得到读出系统数据转化为NP数组"""
    recvData = client.recv(16388, MSG_WAITALL)  # 接收数据
    # frame_valid(recvData)
    byteDATA = recvData[2:16386]  # 截取帧头和帧尾部
    uint32data = numpy.frombuffer(byteDATA[::-1], dtype=numpy.uint32)  # 4字节一读
    uint32data.shape = (64, 64)  # 转为64×64的uint32 array
    uint32data = numpy.rot90(uint32data, 2)
    # sys_log.debug('mean=%.4e, std=%.4e, max=%.4e, min=%.4e' %
    #               (numpy.mean(uint32data), numpy.std(uint32data), numpy.max(uint32data), numpy.min(uint32data)))
    return uint32data


def optical_readout_config(client, v_bias1=1, v_bias2=0, v_on=10, v_off=-10, i_level=100):
    """
    设置读出系统的各个参数
    v_bias1/2:  偏置1/2电压 单位-V
    v_on/v_off: 开启/关闭电压 单位-V
    i_level:    电流档位 单位-nA
    """
    command = 'FF00'  # 命令头
    command = command + numpy.float32(v_bias1).tobytes().hex()  # v_bias1
    command = command + numpy.float32(v_bias2).tobytes().hex()  # v_bias2
    command = command + numpy.float32(v_on).tobytes().hex()  # v_on
    command = command + numpy.float32(v_off).tobytes().hex()  # v_off
    command = command + numpy.uint8(i_level).tobytes().hex()  # i_level
    command = command + '0D0A'  # 命令尾
    send_data = bytes.fromhex(command)  # 转为bytes
    client.send(send_data)  # 发送指令
    sys_log.info('TCP设备配置完成')
    sys_log.debug('Vbias1=%.2f V, Vbias2=%.2f V, Von=%.2f V, Vbias=%.2f V, Ilevel=%d nA' %
                  (v_bias1, v_bias2, v_on, v_off, i_level))


def close_optical_client(client):
    """关闭TCP Client"""
    client.close()
    sys_log.info('TCP Client关闭')
