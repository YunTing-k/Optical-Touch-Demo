import time
import pyautogui as pgui
import numpy
import cv2
import optical_readout
import sys_logger
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt import QtCore
import cap_readout
from VirtualKey import down_up, vk, PressKey, ReleaseKey

sys_log = sys_logger.DemoLogger().logger  # logger


def state_define():  # 定义GUI状态
    """
    IDLE_Capture:    等待 [进行] 录入指纹 (指纹录入后至少等待finger_wait秒才能进入下一操作)
    IDLE_Finger:     录入指纹后等待 [触发] 指纹匹配
    Finger_Match:    [进行] 指纹识别
    Finger_HMI:      指纹识别对应的人机交互
    IDLE_Tap:        等待 [触发] 点击
    Tap_HMI:         点击对应的人机交互
    IDLE_Sweep:      等待 [触发] 轻扫
    Sweep_Detect:    [进行] 扫描方向判断
    Sweep_HMI:       轻扫对应的人机交互
    IDLE_Gesture:    等待 [触发] 手势判断
    Gesture_Detect:  [进行] 手势方向判断
    Gesture_HMI:     手势对应的人机交互
    IDLE_Sign:       等待 [触发] 签字
    Sign_Accumulate: [进行] 签字累加
    Sign_HMI:        签字对应的人机交互
    """
    sate_dic = dict(IDLE_Capture=0, IDLE_Finger=1, Finger_Match=2, Finger_HMI=3, IDLE_Tap=4, Tap_HMI=5, IDLE_Sweep=6,
                    Sweep_Detect=7, Sweep_HMI=8, IDLE_Gesture=9, Gesture_Detect=10, Gesture_HMI=11, IDLE_Sign=12,
                    Sign_Accumulate=13, Sign_HMI=14)
    return sate_dic


def logic_define():  # 定义状态对应的函数和逻辑
    global STATE
    state_logic = {
        STATE['IDLE_Capture']: finger_capture,
        STATE['IDLE_Finger']: finger_trigger_capture,
        STATE['Finger_Match']: finger_match,
        STATE['Finger_HMI']: finger_hmi,
        STATE['IDLE_Tap']: tap_trigger_capture,
        STATE['Tap_HMI']: tap_hmi,
        STATE['IDLE_Sweep']: sweep_trigger_capture,
        STATE['Sweep_Detect']: sweep_detect,
        STATE['Sweep_HMI']: sweep_hmi,
        STATE['IDLE_Gesture']: gesture_trigger_capture,
        STATE['Gesture_Detect']: gesture_detect,
        STATE['Gesture_HMI']: gesture_hmi,
        STATE['IDLE_Sign']: sign_trigger_capture,
        STATE['Sign_Accumulate']: sign_detect,
        STATE['Sign_HMI']: sign_hmi
    }
    return state_logic


def change_vbias3_cap(spin):  # 设置电容屏参数
    global v_bias3_cap, state_timer
    state_timer.stop()
    v_bias3_cap = spin.value()
    state_timer.start(1)


def change_vbias4_cap(spin):  # 设置电容屏参数
    global v_bias4_cap
    state_timer.stop()
    v_bias4_cap = spin.value()
    state_timer.start(1)


def change_von_cap(spin):  # 设置电容屏参数
    global v_on_cap
    state_timer.stop()
    v_on_cap = spin.value()
    state_timer.start(1)


def change_voff_cap(spin):  # 设置电容屏参数
    global v_off_cap
    state_timer.stop()
    v_off_cap = spin.value()
    state_timer.start(1)


def set_cap_param():  # 设置电容屏参数
    global dll, device_handle, v_bias3_cap, v_bias4_cap, v_on_cap, v_off_cap, state_timer
    state_timer.stop()
    cap_readout.ftd_TFT_config(dll, device_handle, v_bias3_cap, v_bias4_cap, v_on_cap, v_off_cap)
    time.sleep(1)
    state_timer.start(1)


def cap_sys_reset():  # 系统复位
    global dll, device_handle, state_timer
    state_timer.stop()
    cap_readout.ftd_reset(dll, device_handle)
    time.sleep(5)
    state_timer.start(1)


def cap_cache_clear():  # 清空缓冲区
    global dll, device_handle, state_timer
    state_timer.stop()
    time.sleep(1)
    cap_readout.ftd_purge(dll, device_handle)
    state_timer.start(1)


def cap_image_preprocess(data):  # 电容屏数据预处理
    max_value = numpy.max(data)
    data = 255 * numpy.divide(data, max_value)
    data = numpy.uint8(data)
    return data


def save_image_cap():  # 图片上采样并保存
    global state_timer
    state_timer.stop()
    data = cap_image_preprocess(cap_readout.ftd_get_img(dll, device_handle))
    save_data = cv2.resize(data, dsize=[512, 512])
    cv2.imwrite('1.png', save_data)
    state_timer.start(1)


def change_vbias1_optical(spin):  # 设置OPD参数
    global v_bias1_optical, state_timer
    state_timer.stop()
    v_bias1_optical = spin.value()
    state_timer.start(1)


def change_vbias2_optical(spin):  # 设置OPD参数
    global v_bias2_optical, state_timer
    state_timer.stop()
    v_bias2_optical = spin.value()
    state_timer.start(1)


def change_von_optical(spin):  # 设置OPD参数
    global v_on_optical, state_timer
    state_timer.stop()
    v_on_optical = spin.value()
    state_timer.start(1)


def change_voff_optical(spin):  # 设置OPD参数
    global v_off_optical, state_timer
    state_timer.stop()
    v_off_optical = spin.value()
    state_timer.start(1)


def change_ilevel_optical(spin):  # 设置OPD参数
    global i_level_optical, state_timer
    state_timer.stop()
    i_level_optical = spin.value()
    state_timer.start(1)


def set_optical_param():  # 设置OPD参数
    global v_bias1_optical, v_bias2_optical, v_on_optical, v_off_optical,\
        i_level_optical, client, stream_start, state_timer
    state_timer.stop()
    optical_readout.stop_stream(client)  # 先停止数据流
    time.sleep(1)
    stream_start = 0
    optical_readout.optical_readout_config(client, v_bias1_optical, v_bias2_optical,
                                           v_on_optical, v_off_optical, i_level_optical)
    time.sleep(1)
    state_timer.start(1)


def get_bright_data():  # 捕获光场数据
    global cali_amount, client, bright_data, client, state_timer, stream_start
    state_timer.stop()
    if not stream_start:
        stream_start = 1
        optical_readout.start_stream(client)
    for i in range(cali_amount):
        bright_data[:, :, i] = optical_readout.optical_data_get(client)
    stream_start = 0
    optical_readout.stop_stream(client)
    sys_log.info('光场捕获成功')
    state_timer.start(1)


def get_dark_data():  # 捕获暗场数据
    global cali_amount, dark_data, client, state_timer, stream_start
    state_timer.stop()
    if not stream_start:
        stream_start = 1
        optical_readout.start_stream(client)
    for i in range(cali_amount):
        dark_data[:, :, i] = optical_readout.optical_data_get(client)
    stream_start = 0
    optical_readout.stop_stream(client)
    sys_log.info('暗场捕获成功')
    state_timer.start(1)


def unity_click_and_press(key):
    global screen_size
    pgui.moveTo(200, 30, duration=0.1)  # 移动鼠标
    time.sleep(0.1)
    pgui.click(button='left')  # 点击
    time.sleep(0.1)
    if key != 'W':
        pgui.keyDown(key)  # 按键触发
        time.sleep(0.1)
        pgui.keyUp(key)  # 按键触发
    else:
        pgui.keyDown(key)  # 按键触发
        time.sleep(1)
        pgui.keyUp(key)  # 按键触发
        pgui.keyDown(key)  # 按键触发
        time.sleep(1)
        pgui.keyUp(key)  # 按键触发


def optical_img_process(data, mode=1):  # 光传感数据预处理
    global min_data, data_sub
    if mode == 1:
        data = 255 * numpy.divide(numpy.subtract(data, min_data), data_sub)
        data[data > 255] = 255
        data[data < 0] = 0
        data = numpy.uint8(data)
    else:
        data = 255 * numpy.divide(numpy.subtract(data, min_data), data_sub) / 4
        data[data > 255] = 255
        data[data < 0] = 0
        data = numpy.uint8(data)
    return data


def save_image_optical():  # 图片上采样并保存
    global state_timer, stream_start, client
    state_timer.stop()
    if not stream_start:
        optical_readout.start_stream(client)
        data = optical_img_process(optical_readout.optical_data_get(client), 2)
        optical_readout.stop_stream(client)
    else:
        data = optical_img_process(optical_readout.optical_data_get(client), 2)
    save_data = cv2.resize(data, dsize=[512, 512])
    cv2.imwrite('2.png', save_data)
    state_timer.start(1)


def finger_capture():  # 捕获手指数据 其他啥也不做
    global data_display, state_timer, dll, device_handle
    state_timer.stop()  # 停止timer防止阻塞
    data = cap_image_preprocess(cap_readout.ftd_get_img(dll, device_handle))
    data_display.setImage(data)  # 更新sensing数据
    state_timer.start(1)  # 启用timer


def capture_image():  # 捕获图像用于指纹匹配(保存按钮按下后就会触发 并触发等待)
    global sample, if_sample_captured, single_shot_timer1, state_timer,\
        finger_wait, captured_bin_display, saveBtn, dll, device_handle
    state_timer.stop()  # 停止timer防止阻塞
    data = cap_image_preprocess(cap_readout.ftd_get_img(dll, device_handle))
    _, sample = cv2.threshold(data, 127, 255, cv2.THRESH_BINARY_INV)  # 转二值
    captured_bin_display.setImage(sample)
    if_sample_captured = True
    sys_log.info('样本捕获完成')
    unity_click_and_press('0')
    saveBtn.setEnabled(False)  # 捕获后的等待就禁用保存按钮
    single_shot_timer1.start(1000 * finger_wait)  # 定时延迟触发
    state_timer.start(1)  # 启用timer


def finger_trigger_capture():  # 捕获数据 并且判断是否触发指纹匹配
    global data_display, finger_thresh, current_state, STATE, state_timer, dll, device_handle
    state_timer.stop()  # 停止timer防止阻塞
    data = cap_image_preprocess(cap_readout.ftd_get_img(dll, device_handle))
    data_display.setImage(data)  # 更新sensing数据
    _, bin_data = cv2.threshold(data, 127, 1, cv2.THRESH_BINARY_INV)  # 转二值
    if numpy.sum(bin_data) > (numpy.size(bin_data) * finger_thresh):
        current_state = STATE['Finger_Match']  # 记录指纹后并且激励的像素数目超过预定的比例 切换到Finger Match状态
        sys_log.debug('开始匹配指纹')
    state_timer.start(1)  # 启用timer


def finger_match():  # 使用SIFT算子进行指纹匹配
    global finger_bin_display, sample, current_state, STATE, sift, finger_match_display, dll, device_handle
    state_timer.stop()  # 停止timer防止阻塞
    # 再次捕获数据
    data = cap_image_preprocess(cap_readout.ftd_get_img(dll, device_handle))
    # 转为二值图像
    _, bin_data = cv2.threshold(data, 127, 255, cv2.THRESH_BINARY_INV)  # 转二值
    sample_keypoint, sample_descr = sift.detectAndCompute(sample, None)  # 指纹样本的SIFT计算
    keypoint, descriptor = sift.detectAndCompute(bin_data, None)  # 传入图像的特征点和描述
    matches = flann.knnMatch(sample_descr, descriptor, k=2)  # 使用KNN匹配以匹配特征点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    if len(good_matches) >= match_thresh:  # 匹配进入状态
        current_state = STATE['Finger_HMI']
        sys_log.debug('匹配成功')
        result_image = cv2.drawMatches(sample, sample_keypoint, bin_data, keypoint, good_matches, None, flags=2)
        cv2.imwrite('C:/1.png', cv2.resize(bin_data, dsize=[512, 512]))
        finger_match_display.setImage(result_image)
    else:  # 匹配失败 回到状态IDLE_Finger
        current_state = STATE['IDLE_Finger']
        sys_log.warning('匹配失败')
    finger_bin_display.setImage(bin_data)
    state_timer.start(1)  # 启用timer


def finger_hmi():  # HMI的人机交互
    global state_timer, current_state, STATE, tap_wait, data_display, device_handle, dll, screen_size
    state_timer.stop()  # 停止timer防止阻塞
    data_display.setImage(numpy.zeros([cap_size[0], cap_size[1]]))  # 更新sensing数据
    current_state = STATE['IDLE_Tap']  # 到IDLE_Tap
    cap_readout.ftd_reset(dll, device_handle)  # 重置系统
    cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
    unity_click_and_press('F')
    state_timer.start(1000 * tap_wait)  # 启用timer 并等待固定时间


def tap_trigger_capture():
    global data_display, current_state, STATE, state_timer, tap_trigger_thresh,\
        tap_pos, tap_bin_display, dll, device_handle
    state_timer.stop()  # 停止timer防止阻塞
    data = cap_image_preprocess(cap_readout.ftd_get_img(dll, device_handle))
    data_display.setImage(data)  # 更新sensing数据
    _, bin_data = cv2.threshold(data, 127, 1, cv2.THRESH_BINARY_INV)  # 转二值
    tap_bin_display.setImage(bin_data)  # 更新二值绘图
    if numpy.sum(bin_data) > (numpy.size(bin_data) * tap_trigger_thresh):  # 超过门限 判定位点击
        tap_pos = numpy.mean(numpy.argwhere(bin_data > 0), axis=0)  # 返回得到触发时的位置
        update_tap_pos_graph()
        sys_log.info('检测到点击')
        current_state = STATE['Tap_HMI']  # 点击完成 切换到Tap_HMI状态
    state_timer.start(1)  # 启用timer


def update_tap_pos_graph():
    global tap_line_x, tap_line_y, tap_pos
    tap_line_x.setPos(tap_pos[1])
    tap_line_y.setPos(tap_pos[0])


def tap_hmi():
    global state_timer, current_state, STATE, sweep_wait, dll, device_handle
    state_timer.stop()  # 停止timer防止阻塞
    current_state = STATE['IDLE_Sweep']  # 到IDLE_Sweep
    cap_readout.ftd_reset(dll, device_handle)  # 重置系统
    cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
    unity_click_and_press('C')
    state_timer.start(1000 * sweep_wait)  # 启用timer


def sweep_trigger_capture():  # 捕获数据 并且判断是否触发轻扫
    global data_display, current_state, STATE, state_timer, sweep_trigger_thresh, sweep_first_pos,\
        sweep_current_pos, dll, device_handle
    state_timer.stop()  # 停止timer防止阻塞
    data = cap_image_preprocess(cap_readout.ftd_get_img(dll, device_handle))
    data_display.setImage(data)  # 更新sensing数据
    _, bin_data = cv2.threshold(data, 127, 1, cv2.THRESH_BINARY_INV)  # 转二值
    sweep_bin_display.setImage(bin_data)  # 更新二值绘图
    if numpy.sum(bin_data) > (numpy.size(bin_data) * sweep_trigger_thresh):
        sweep_first_pos = numpy.mean(numpy.argwhere(bin_data > 0), axis=0)  # 返回得到触发时的位置
        sweep_current_pos = sweep_first_pos
        update_sweep_current_graph()
        update_sweep_direction_graph()
        sys_log.debug('开始检测轻扫')
        current_state = STATE['Sweep_Detect']  # 激励的像素数目超过预定的比例 切换到Sweep_Detect状态
    state_timer.start(1)  # 启用timer


def update_sweep_current_graph():  # 更新目前sweep位置
    global sweep_current_pos, sweep_line_x, sweep_line_y
    sweep_line_x.setPos(sweep_current_pos[1])
    sweep_line_y.setPos(sweep_current_pos[0])


def update_sweep_direction_graph():  # 更新目前方向
    global sweep_pos, sweep_current_pos, sweep_first_pos, sweep_direction_graph, cap_size
    x_vec = (sweep_current_pos[1] - sweep_first_pos[1]) / cap_size[1]
    y_vec = (sweep_current_pos[0] - sweep_first_pos[0]) / cap_size[0]
    sweep_pos = numpy.array([[0, 0], [x_vec, y_vec]])  # 计算点的位置
    sweep_direction_graph.setData(pos=sweep_pos, adj=sweep_adj, pen=sweep_lines, size=sweep_size, symbol=sweep_symbols,
                                  pxMode=False)


def sweep_detect():  # 捕获数据 并且判断是否结束触发
    global data_display, current_state, STATE, state_timer, sweep_quit_thresh,\
        sweep_last_pos, sweep_current_pos, device_handle, dll
    state_timer.stop()  # 停止timer防止阻塞
    data = cap_image_preprocess(cap_readout.ftd_get_img(dll, device_handle))
    data_display.setImage(data)  # 更新sensing数据
    _, bin_data = cv2.threshold(data, 127, 1, cv2.THRESH_BINARY_INV)  # 转二值
    sweep_bin_display.setImage(bin_data)  # 更新二值绘图
    sweep_current_pos = numpy.mean(numpy.argwhere(bin_data > 0), axis=0)  # 现在的位置
    if 0 < numpy.sum(bin_data) < (numpy.size(bin_data) * sweep_quit_thresh):
        sweep_last_pos = sweep_current_pos  # 返回即将退出触发时的位置
        current_state = STATE['Sweep_HMI']  # 激励的像素数目小于预定的比例 切换到Sweep_HMI状态
        sys_log.info('轻扫结束')
    else:
        update_sweep_current_graph()
        update_sweep_direction_graph()
    state_timer.start(1)  # 启用timer


def sweep_direction_judge():  # 判断sweep方向 0-上 1-左 2-右 3-下
    global sweep_first_pos, sweep_last_pos
    x = sweep_last_pos[1] - sweep_first_pos[1]  # x矢量s
    y = sweep_last_pos[0] - sweep_first_pos[0]  # y矢量
    angle = 180 * numpy.arctan(y/x) / numpy.pi
    sys_log.debug(('x=%d, y=%d' % (x, y)))
    if x > 0:  # 有右滑分量
        if numpy.abs(angle) <= 45:  # 右滑
            return 2
        elif angle > 45:  # 下滑
            return 3
        else:  # 上滑
            return 0
    elif x < 0:  # 有左滑分量
        if numpy.abs(angle) <= 45:  # 左滑
            return 1
        elif angle > 45:  # 上滑
            return 0
        else:  # 下滑
            return 3


def sweep_hmi():  # HMI的人机交互
    global state_timer, current_state, STATE, tap_wait, client, gesture_wait, current_sweep,\
        sweep_list, sweep_amount, single_shot_timer2, before_sign
    state_timer.stop()  # 停止timer防止阻塞
    if (current_sweep + 1) <= sweep_amount:  # 没有扫完
        direction = sweep_direction_judge()  # 扫描方向
        sys_log.debug(('Sweep Direction= ' + str(direction)))
        sys_log.debug('Current Sweep= ' + str(current_sweep))
        if direction == sweep_list[current_sweep]:  # 方向吻合
            unity_click_and_press(sweep_key[current_sweep])
            current_sweep += 1
        else:  # 方向错误
            pass
        if (current_sweep + 1) > sweep_amount:  # 没有扫完
            sys_log.debug('轻扫结束')
            current_state = STATE['IDLE_Gesture']  # 回到IDLE_Gesture
            cap_readout.ftd_reset(dll, device_handle)  # 重置系统
            cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
            state_timer.start(1)  # 启用timer
            single_shot_timer2.start(1000 * (1 + before_sign))  # 启用timer
        else:
            current_state = STATE['IDLE_Sweep']  # 回到IDLE_Sweep
            cap_readout.ftd_reset(dll, device_handle)  # 重置系统
            cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
            state_timer.start(1000 * gesture_wait)  # 启用timer
    else:  # 扫完了
        current_sweep += 1
        sys_log.debug('轻扫结束')
        current_state = STATE['IDLE_Gesture']  # 回到IDLE_Gesture
        cap_readout.ftd_reset(dll, device_handle)  # 重置系统
        cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
        state_timer.start(1)  # 启用timer
        single_shot_timer2.start(1000 * (1 + before_sign))  # 启用timer


def gesture_trigger_capture():  # 捕获数据 并且判断是否触发手势
    global data_display, current_state, STATE, state_timer, gesture_trigger_thresh, gesture_first_pos,\
        gesture_current_pos, client, stream_start
    state_timer.stop()  # 停止timer防止阻塞
    # 捕获数据
    if not stream_start:
        stream_start = 1
        optical_readout.start_stream(client)
    data = optical_img_process(optical_readout.optical_data_get(client))
    data_display.setImage(data)  # 更新sensing数据
    _, bin_data = cv2.threshold(data, 127, 1, cv2.THRESH_BINARY_INV)  # 转二值
    gesture_bin_display.setImage(bin_data)  # 更新二值绘图
    if numpy.sum(bin_data) > (numpy.size(bin_data) * gesture_trigger_thresh):
        gesture_first_pos = numpy.mean(numpy.argwhere(bin_data > 0), axis=0)  # 返回得到触发时的位置
        gesture_current_pos = gesture_first_pos
        update_gesture_current_graph()
        update_gesture_direction_graph()
        sys_log.debug('开始检测手势')
        current_state = STATE['Gesture_Detect']  # 激励的像素数目超过预定的比例 切换到Gesture_Detect状态
    state_timer.start(1)  # 启用timer


def update_gesture_current_graph():  # 更新目前gesture位置
    global gesture_current_pos, gesture_line_x, gesture_line_y
    gesture_line_x.setPos(gesture_current_pos[1])
    gesture_line_y.setPos(gesture_current_pos[0])


def update_gesture_direction_graph():  # 更新目前gesture方向
    global gesture_pos, gesture_current_pos, gesture_first_pos, gesture_direction_graph, optical_size
    x_vec = (gesture_current_pos[1] - gesture_first_pos[1]) / optical_size[1]
    y_vec = (gesture_current_pos[0] - gesture_first_pos[0]) / optical_size[0]
    gesture_pos = numpy.array([[0, 0], [x_vec, y_vec]])  # 计算点的位置
    gesture_direction_graph.setData(pos=gesture_pos, adj=gesture_adj, pen=gesture_lines, size=gesture_size,
                                    symbol=gesture_symbols, pxMode=False)


def gesture_detect():  # 捕获数据 并且判断是否结束触发
    global data_display, current_state, STATE, state_timer, gesture_quit_thresh,\
        gesture_last_pos, gesture_current_pos, client, stream_start
    state_timer.stop()  # 停止timer防止阻塞
    # 捕获数据
    if not stream_start:
        stream_start = 1
        optical_readout.start_stream(client)
    data = optical_img_process(optical_readout.optical_data_get(client))
    data_display.setImage(data)  # 更新sensing数据
    _, bin_data = cv2.threshold(data, 127, 1, cv2.THRESH_BINARY_INV)  # 转二值
    gesture_bin_display.setImage(bin_data)  # 更新二值绘图
    gesture_current_pos = numpy.mean(numpy.argwhere(bin_data > 0), axis=0)  # 现在的位置
    if 0 < numpy.sum(bin_data) < (numpy.size(bin_data) * gesture_quit_thresh):
        gesture_last_pos = gesture_current_pos  # 返回即将退出触发时的位置
        current_state = STATE['Gesture_HMI']  # 激励的像素数目小于预定的比例 切换到Gesture_HMI状态
        sys_log.info('手势结束')
    else:
        update_gesture_current_graph()
        update_gesture_direction_graph()
    state_timer.start(1)  # 启用timer


def gesture_direction_judge():  # 判断gesture方向 0-上 1-左 2-右 3-下
    global gesture_first_pos, gesture_last_pos
    x = gesture_last_pos[1] - gesture_first_pos[1]  # x矢量
    y = gesture_last_pos[0] - gesture_first_pos[0]  # y矢量
    angle = 180 * numpy.arctan(y/x) / numpy.pi
    if x > 0:  # 有右滑分量
        if numpy.abs(angle) <= 45:  # 右滑
            return 2
        elif angle > 45:  # 下滑
            return 3
        else:  # 上滑
            return 0
    elif x < 0:  # 有左滑分量
        if numpy.abs(angle) <= 45:  # 左滑
            return 1
        elif angle > 45:  # 上滑
            return 0
        else:  # 下滑
            return 3


def gesture_hmi():  # HMI的人机交互
    global state_timer, current_state, STATE, sign_wait, client, stream_start, gesture_key
    state_timer.stop()  # 停止timer防止阻塞
    # stream_start = 0
    # optical_readout.stop_stream(client)
    direction = gesture_direction_judge()
    sys_log.debug(('Gesture Direction' + str(direction)))
    unity_click_and_press(gesture_key[direction])
    current_state = STATE['IDLE_Gesture']
    state_timer.start(1000 * sign_wait)  # 启用timer


def sign_trigger_capture():  # 捕获数据 并且判断是否触发签名
    global data_display, current_state, STATE, state_timer, sign_trigger_thresh, sign_first_pos,\
        sign_current_pos, client, stream_start
    state_timer.stop()  # 停止timer防止阻塞
    # 捕获数据
    if not stream_start:
        stream_start = 1
        optical_readout.start_stream(client)
    data = optical_img_process(optical_readout.optical_data_get(client), 2)
    data = data[3:63, 3:63]
    data_display.setImage(data)  # 更新sensing数据
    _, bin_data = cv2.threshold(data, 254, 1, cv2.THRESH_BINARY)  # 转二值
    sign_bin_display.setImage(bin_data)  # 更新二值绘图
    if numpy.sum(bin_data) > (numpy.size(bin_data) * sign_trigger_thresh):
        sign_first_pos = numpy.mean(numpy.argwhere(bin_data > 0), axis=0)  # 返回得到触发时的位置
        sign_current_pos = sign_first_pos
        update_sign_current_graph()
        update_sign_accumulate_graph(bin_data)
        sys_log.debug('开始签字')
        current_state = STATE['Sign_Accumulate']  # 激励的像素数目超过预定的比例 切换到Sign_Accumulate状态
    state_timer.start(1)  # 启用timer


def update_sign_current_graph():  # 更新目前sign位置
    global sign_current_pos, sign_line_x, sign_line_y
    sign_line_x.setPos(sign_current_pos[1])
    sign_line_y.setPos(sign_current_pos[0])


def update_sign_accumulate_graph(bin_data):  # 累加sign
    global sign_img
    add_img = sign_img.astype('int64') + bin_data.astype('int64')
    sign_img[add_img > 0] = 255
    sign_accumulate_display.setImage(sign_img)


def sign_detect():  # 捕获数据 并且判断是否结束触发
    global data_display, current_state, STATE, state_timer, sign_quit_thresh,\
        sign_current_pos, client, stream_start, sign_img
    state_timer.stop()  # 停止timer防止阻塞
    # 捕获数据
    if not stream_start:
        stream_start = 1
        optical_readout.start_stream(client)
    data = optical_img_process(optical_readout.optical_data_get(client), 2)
    data = data[3:63, 3:63]
    data_display.setImage(data)  # 更新sensing数据
    _, bin_data = cv2.threshold(data, 254, 1, cv2.THRESH_BINARY)  # 转二值
    sign_bin_display.setImage(bin_data)  # 更新二值绘图
    sign_current_pos = numpy.mean(numpy.argwhere(bin_data > 0), axis=0)  # 现在的位置
    if 0 < numpy.sum(bin_data) < (numpy.size(bin_data) * sign_quit_thresh):
        cv2.imwrite('C:/2.png', cv2.resize(sign_img, dsize=[512, 512]))
        current_state = STATE['Sign_HMI']  # 激励的像素数目小于预定的比例 切换到Sign_HMI状态
        sys_log.info('签字结束')
    else:
        update_sign_current_graph()
        update_sign_accumulate_graph(bin_data)
    state_timer.start(1)  # 启用timer


def sign_hmi():
    global state_timer, stream_start, client, return_wait, current_state, STATE
    state_timer.stop()  # 停止timer防止阻塞
    unity_click_and_press('3')
    unity_click_and_press('N')
    stream_start = 0
    optical_readout.stop_stream(client)
    current_state = STATE['IDLE_Capture']
    state_timer.start(1000 * return_wait)


def state_transfer():  # 状态机状态转移函数
    global current_state, LOGIC, state_label
    state_label.setText('Current State:' + str(current_state))
    LOGIC[current_state]()


def get_cali_param():  # 计算标定参数
    global bright_data, dark_data, state_timer, max_data, min_data, data_sub
    state_timer.stop()
    max_data = numpy.max(bright_data, axis=2)
    min_data = numpy.min(dark_data, axis=2)
    data_sub = numpy.subtract(max_data, min_data)
    set_unscale_index = numpy.where(data_sub == 0)  # 差为0，不进行scale
    data_sub[set_unscale_index] = 1
    sys_log.info('标定完成')
    state_timer.start(1)


def goto_IDLE_Capture():  # 跳转到IDLE_Capture状态
    global current_state, saveBtn
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    saveBtn.setEnabled(True)  # 允许使用按钮
    current_state = STATE['IDLE_Capture']  # 切换状态
    cap_readout.ftd_reset(dll, device_handle)  # 重置系统
    cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
    time.sleep(1)
    state_timer.start(1)


def goto_Finger_HMI():  # 跳转到Finger_HMI状态
    global current_state
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['Finger_HMI']  # 切换状态
    state_timer.start(1)


def goto_IDLE_Finger():  # 跳转到IDLE_Finger状态
    global current_state
    current_state = STATE['IDLE_Finger']  # 切换状态


def goto_IDLE_Sweep():  # 跳转到IDLE_Sweep状态
    global current_state
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['IDLE_Sweep']  # 切换状态
    cap_readout.ftd_reset(dll, device_handle)  # 重置系统
    cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
    time.sleep(1)
    state_timer.start(1)


def goto_Sweep_HMI():  # 跳转到Sweep_HMI状态
    global current_state
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['Sweep_HMI']  # 切换状态
    state_timer.start(1)


def goto_IDLE_Tap():  # 跳转到IDLE_Tap状态
    global current_state
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['IDLE_Tap']  # 切换状态
    cap_readout.ftd_reset(dll, device_handle)  # 重置系统
    cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
    time.sleep(1)
    state_timer.start(1)


def goto_Tap_HMI():  # 跳转到Tap_HMI状态
    global current_state
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['Tap_HMI']  # 切换状态
    state_timer.start(1)


def goto_IDLE_Gesture():  # 跳转到IDLE_Gesture状态
    global current_state
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['IDLE_Gesture']  # 切换状态
    cap_readout.ftd_reset(dll, device_handle)  # 重置系统
    cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
    time.sleep(1)
    single_shot_timer2.start(1000 * (gesture_wait + before_sign))  # 启用timer
    state_timer.start(1)


def goto_Gesture_HMI():  # 跳转到Gesture_HMI状态
    global current_state
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['Gesture_HMI']  # 切换状态
    state_timer.start(1)


def goto_IDLE_Sign():  # 跳转到IDLE_Sign状态
    global current_state, sign_img, sign_accumulate_display
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['IDLE_Sign']  # 切换状态
    cap_readout.ftd_reset(dll, device_handle)  # 重置系统
    cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
    sign_img = numpy.zeros((optical_size[0] - 4, optical_size[1] - 4), dtype='uint8')  # 签字
    sign_accumulate_display.setImage(sign_img)
    time.sleep(1)
    state_timer.start(1)


def goto_Sign_HMI():  # 跳转到Sign_HMI状态
    global current_state
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    current_state = STATE['Sign_HMI']  # 切换状态
    state_timer.start(1)


def goto_scene_0():  # 跳到场景0
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    unity_click_and_press('0')
    state_timer.start(1)


def goto_scene_1():  # 跳到场景1
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    unity_click_and_press('1')
    state_timer.start(1)


def goto_scene_2():  # 跳到场景2
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    unity_click_and_press('2')
    state_timer.start(1)


def goto_scene_3():  # 跳到场景3
    state_timer.stop()
    single_shot_timer1.stop()
    single_shot_timer2.stop()
    unity_click_and_press('3')
    state_timer.start(1)


if __name__ == '__main__':
    """定义触摸屏下位机"""
    # 打开DLL并创建handle
    dll = cap_readout.create_ftd_dll()
    device_handle = cap_readout.ftd_handle(dll)  # 创建handle
    # 系统复位
    cap_readout.ftd_reset(dll, device_handle)
    # 配置系统
    cap_readout.ftd_purge(dll, device_handle)  # 清除缓存
    cap_readout.ftd_set_bit_mode(dll, device_handle)  # 设置启动模式
    cap_readout.ftd_set_usb_param(dll, device_handle)  # 设置传输大小
    cap_readout.ftd_set_flow_ctr(dll, device_handle)  # 设置流量控制
    cap_readout.ftd_set_timeout(dll, device_handle)  # 设置超时
    # 配置TFT
    # cap_readout.ftd_TFT_config(dll, device_handle)
    time.sleep(1)
    v_bias3_cap = 3
    v_bias4_cap = -5
    v_on_cap = 10
    v_off_cap = -10
    """定义光传感下位机"""
    stream_start = 0  # 数据流开始
    client = optical_readout.create_tcp_client()  # 创建一个TCP Client (一开始就连接读出系统)
    v_bias1_optical = 1
    v_bias2_optical = 0
    v_on_optical = 10
    v_off_optical = -10
    i_level_optical = 100

    """全局参数"""
    screen_size = [2560, 1440]  # 副屏尺寸
    display_size = [500, 500]  # 展示数据的尺寸 width, height
    state_panel_width = 100  # 控制面板的宽度
    cap_panel_width = 100  # 控制面板的宽度
    STATE = state_define()  # 状态机的所有状态列表
    LOGIC = logic_define()  # 状态机对应的逻辑函数
    current_state = STATE['IDLE_Capture']  # 当前状态
    sweep_list = [0, 1, 2, 3]  # 上0 左1 右2 下3
    sweep_key = ['I', 'J', 'L', 'K']
    gesture_key = ['W', 'A', 'D', 'S']
    current_sweep = 0
    sweep_amount = len(sweep_list)
    gesture_amount = 3  # 挥动次数
    current_gesture = 0  # 已经挥动的次数
    return_wait = 5  # 返回时间

    """算法参数"""
    cap_size = [256, 256]  # 触控屏幕数据大小
    optical_size = [64, 64]  # 光响应数据大小
    cali_amount = 50  # 标定采样次数
    bright_data = numpy.ones([optical_size[0], optical_size[1], cali_amount])  # 亮场数据
    min_data = numpy.ones([optical_size[0], optical_size[1]])  # 最暗数据
    max_data = 2 * numpy.ones([optical_size[0], optical_size[1]])  # 最亮数据
    data_sub = numpy.ones([optical_size[0], optical_size[1]])  # 最亮数据 - 最暗数据
    dark_data = numpy.zeros([optical_size[0], optical_size[1], cali_amount])  # 暗场数据

    finger_wait = 2  # 指纹录入后需要等待时间
    finger_thresh = 0.04  # 触发指纹识别的门限
    if_sample_captured = False  # 是否采样指纹标志
    sample = numpy.zeros((cap_size[0], cap_size[1], 3), dtype='uint8')  # 指纹采样
    match_thresh = 10  # 指纹匹配点数阈值
    sift = cv2.SIFT_create()  # SIFT检测器
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})  # FLANN匹配器

    sweep_wait = 5  # 指纹交互后等待时间
    sweep_trigger_thresh = 0.02  # 触发轻扫判断的门限
    sweep_quit_thresh = 0.01  # 触发结束轻扫判断的门限
    sweep_first_pos = [0, 0]  # 第一次触发时的绝对位置
    sweep_current_pos = [0, 0]  # 轻扫触发的现在的绝对位置
    sweep_last_pos = [0, 0]  # 最后一次触发时的绝对位置

    tap_wait = 5  # 轻扫交互后等待时间
    tap_trigger_thresh = 0.03  # 触发点击判断的门限
    tap_pos = [0, 0]  # 点击位置

    gesture_wait = 2  # 点击交互后等待时间
    gesture_trigger_thresh = 0.2
    gesture_quit_thresh = 0.15
    gesture_first_pos = [0, 0]  # 第一次触发时的绝对位置
    gesture_current_pos = [0, 0]  # 轻扫触发的现在的绝对位置
    gesture_last_pos = [0, 0]  # 最后一次触发时的绝对位置
    before_sign = 80  # 手势等待时间，超过80s就是直接跳转到sign

    sign_wait = 0.5  # 签名交互后等待时间
    sign_trigger_thresh = 0.001
    sign_quit_thresh = 0.0005
    sign_img = numpy.zeros((optical_size[0] - 4, optical_size[1] - 4), dtype='uint8')  # 签字
    sign_first_pos = [0, 0]  # 第一次触发时的绝对位置
    sign_current_pos = [0, 0]  # 现在的绝对位置
    """窗口构建和全局初始化"""
    app = pg.mkQApp("GUI")  # 创建APP
    pg.setConfigOptions(antialias=True)  # 抗锯齿
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    win = QtWidgets.QMainWindow()  # 主窗口
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(display_size[0] * 2 + state_panel_width + cap_panel_width, display_size[1])
    win.setWindowTitle('GUI Interface')  # 设置标题
    sub_win1 = pg.GraphicsLayoutWidget(show=True)  # 传感信息绘图
    sub_win2 = pg.GraphicsLayoutWidget(show=True)  # 指纹交互绘图
    sub_win3 = pg.GraphicsLayoutWidget(show=True)  # 轻扫交互绘图
    sub_win4 = pg.GraphicsLayoutWidget(show=True)  # 点击交互绘图
    sub_win5 = pg.GraphicsLayoutWidget(show=True)  # 手势交互绘图
    sub_win6 = pg.GraphicsLayoutWidget(show=True)  # 签字交互绘图
    pg.setConfigOptions(imageAxisOrder='row-major')  # 行优先绘图

    """Dock构建"""
    d1 = Dock("Sensing Data", size=(display_size[0], display_size[1]))  # 传感数据绘制 总是尽可能实时显示传感结果
    d2 = Dock("Finger", size=(display_size[0], display_size[1]))  # 指纹交互窗口
    d3 = Dock("Sweep", size=(display_size[0], display_size[1]))  # 轻扫交互窗口
    d4 = Dock("Tap", size=(display_size[0], display_size[1]))  # 点击交互窗口
    d5 = Dock("Gesture", size=(display_size[0], display_size[1]))  # 手势交互窗口
    d6 = Dock("Sign", size=(display_size[0], display_size[1]))  # 签字交互窗口
    d7 = Dock("State Machine", size=(state_panel_width, display_size[1]))  # 状态机控制按钮
    d8 = Dock("Cap Control", size=(state_panel_width, display_size[1]))  # 电容屏读出系统控制按钮
    d9 = Dock("Optical Control", size=(state_panel_width, display_size[1]))  # TCP控制按钮
    """Dock布局"""
    area.addDock(d1, 'left')
    area.addDock(d7, 'right')
    area.addDock(d8, 'right')
    area.addDock(d9, 'right')
    area.addDock(d6, 'right', d1)
    area.addDock(d5, 'above', d6)
    area.addDock(d4, 'above', d5)
    area.addDock(d3, 'above', d4)
    area.addDock(d2, 'above', d3)

    """Dock1 传感信息"""
    data_plot = sub_win1.addPlot()  # 绘制数据的窗口
    data_plot.invertY(True)  # 反转Y轴
    data_plot.setMouseEnabled(x=False, y=False)
    data_plot.showAxis('left', show=False)
    data_plot.showAxis('bottom', show=False)
    data_display = pg.ImageItem()  # 原始数据
    data_plot.addItem(data_display)
    data_plot.setAspectLocked(lock=True)
    d1.addWidget(sub_win1)

    """Dock2 指纹识别交互"""
    # 绘制二值图像的窗口
    finger_bin_plot = sub_win2.addPlot()
    finger_bin_plot.invertY(True)  # 反转Y轴
    finger_bin_plot.setMouseEnabled(x=False, y=False)
    finger_bin_plot.showAxis('left', show=False)
    finger_bin_plot.showAxis('bottom', show=False)
    finger_bin_display = pg.ImageItem()  # 二值图像
    finger_bin_plot.addItem(finger_bin_display)
    finger_bin_plot.setAspectLocked(lock=True)
    # 绘制存储的图像的窗口
    captured_bin_plot = sub_win2.addPlot()
    captured_bin_plot.invertY(True)  # 反转Y轴
    captured_bin_plot.setMouseEnabled(x=False, y=False)
    captured_bin_plot.showAxis('left', show=False)
    captured_bin_plot.showAxis('bottom', show=False)
    captured_bin_display = pg.ImageItem()  # 二值图像
    captured_bin_plot.addItem(captured_bin_display)
    captured_bin_plot.setAspectLocked(lock=True)
    sub_win2.nextRow()
    # 存储匹配结果的窗口
    finger_match_plot = sub_win2.addPlot(colspan=2)
    finger_match_plot.invertY(True)  # 反转Y轴
    finger_match_plot.setMouseEnabled(x=False, y=False)
    finger_match_plot.showAxis('left', show=False)
    finger_match_plot.showAxis('bottom', show=False)
    finger_match_display = pg.ImageItem()  # 匹配结果
    finger_match_plot.addItem(finger_match_display)
    finger_match_plot.setAspectLocked(lock=True)
    # 添加绘图组件到dock中
    d2.addWidget(sub_win2)

    """Dock3 轻扫交互"""
    # 绘制二值图像的窗口
    sweep_bin_plot = sub_win3.addPlot()
    sweep_bin_plot.invertY(True)  # 反转Y轴
    sweep_bin_plot.setMouseEnabled(x=False, y=False)
    sweep_bin_plot.showAxis('left', show=False)
    sweep_bin_plot.showAxis('bottom', show=False)
    sweep_bin_display = pg.ImageItem()  # 二值图像
    sweep_bin_plot.addItem(sweep_bin_display)
    sweep_bin_plot.setAspectLocked(lock=True)
    # 绘制位点、参考线的窗口
    sweep_pos_view = sub_win3.addPlot()
    sweep_pos_view.invertY(True)  # 反转Y轴
    sweep_pos_view.setMouseEnabled(x=False, y=False)
    sweep_pos_view.setXRange(1, cap_size[0])
    sweep_pos_view.setYRange(1, cap_size[1])
    sweep_pos_view.showAxis('left', show=True)
    sweep_pos_view.showAxis('bottom', show=True)
    sweep_pos_view.showAxis('right', show=True)
    sweep_pos_view.showAxis('top', show=True)
    sweep_pos_view.setAspectLocked(lock=True)
    sweep_line_x = pg.InfiniteLine(movable=False, angle=90, label='x={value:0.2f}',
                                   pen=pg.mkPen(color=(200, 200, 100), width=5),
                                   labelOpts={'position': 0.3, 'color': (200, 200, 100),
                                              'fill': (200, 200, 200, 50), 'movable': True})
    sweep_line_y = pg.InfiniteLine(movable=False, angle=0, label='y={value:0.2f}',
                                   pen=pg.mkPen(color=(200, 200, 100), width=5),
                                   labelOpts={'position': 0.3, 'color': (200, 200, 100),
                                              'fill': (200, 200, 200, 50), 'movable': True})
    sweep_pos_view.addItem(sweep_line_x)
    sweep_pos_view.addItem(sweep_line_y)
    sub_win3.nextRow()
    # 绘制方向的窗口
    sweep_direction_view = sub_win3.addPlot(colspan=2)
    sweep_direction_view.invertY(True)  # 反转Y轴
    sweep_direction_view.setMouseEnabled(x=False, y=False)
    sweep_direction_view.setXRange(-1, 1)
    sweep_direction_view.setYRange(-1, 1)
    sweep_direction_view.showAxis('left', show=True)
    sweep_direction_view.showAxis('bottom', show=True)
    sweep_direction_view.showAxis('right', show=True)
    sweep_direction_view.showAxis('top', show=True)
    sweep_direction_view.setAspectLocked(lock=True)
    sweep_direction_graph = pg.GraphItem()
    sweep_pos = numpy.array([[0, 0], [0, 0]])  # 点的位置
    sweep_adj = numpy.array([[0, 1]])
    sweep_lines = numpy.array([(0, 0, 0, 255, 2)],
                              dtype=[('red', numpy.ubyte), ('green', numpy.ubyte),
                              ('blue', numpy.ubyte), ('alpha', numpy.ubyte), ('width', float)])
    sweep_symbols = ['o', '+']
    sweep_size = [0.2, 0.2]
    sweep_direction_graph.setData(pos=sweep_pos, adj=sweep_adj, pen=sweep_lines, size=sweep_size, symbol=sweep_symbols,
                                  pxMode=False)
    sweep_direction_view.addItem(sweep_direction_graph)
    # 添加绘图组件到dock中
    d3.addWidget(sub_win3)

    """Dock4 点击交互"""
    # 绘制二值图像的窗口
    tap_bin_plot = sub_win4.addPlot()
    tap_bin_plot.invertY(True)  # 反转Y轴
    tap_bin_plot.setMouseEnabled(x=False, y=False)
    tap_bin_plot.showAxis('left', show=False)
    tap_bin_plot.showAxis('bottom', show=False)
    tap_bin_display = pg.ImageItem()  # 二值图像
    tap_bin_plot.addItem(tap_bin_display)
    tap_bin_plot.setAspectLocked(lock=True)
    # 绘制位点、参考线的窗口
    tap_pos_view = sub_win4.addPlot()
    tap_pos_view.invertY(True)  # 反转Y轴
    tap_pos_view.setMouseEnabled(x=False, y=False)
    tap_pos_view.setXRange(1, cap_size[0])
    tap_pos_view.setYRange(1, cap_size[1])
    tap_pos_view.showAxis('left', show=True)
    tap_pos_view.showAxis('bottom', show=True)
    tap_pos_view.showAxis('right', show=True)
    tap_pos_view.showAxis('top', show=True)
    tap_pos_view.setAspectLocked(lock=True)
    tap_line_x = pg.InfiniteLine(movable=False, angle=90, label='x={value:0.2f}',
                                 pen=pg.mkPen(color=(200, 200, 100), width=5),
                                 labelOpts={'position': 0.3, 'color': (200, 200, 100),
                                            'fill': (200, 200, 200, 50), 'movable': True})
    tap_line_y = pg.InfiniteLine(movable=False, angle=0, label='y={value:0.2f}',
                                 pen=pg.mkPen(color=(200, 200, 100), width=5),
                                 labelOpts={'position': 0.3, 'color': (200, 200, 100),
                                            'fill': (200, 200, 200, 50), 'movable': True})
    tap_pos_view.addItem(tap_line_x)
    tap_pos_view.addItem(tap_line_y)
    # 添加绘图组件到dock中
    d4.addWidget(sub_win4)

    """Dock5 手势交互"""
    # 绘制二值图像的窗口
    gesture_bin_plot = sub_win5.addPlot()
    gesture_bin_plot.invertY(True)  # 反转Y轴
    gesture_bin_plot.setMouseEnabled(x=False, y=False)
    gesture_bin_plot.showAxis('left', show=False)
    gesture_bin_plot.showAxis('bottom', show=False)
    gesture_bin_display = pg.ImageItem()  # 二值图像
    gesture_bin_plot.addItem(gesture_bin_display)
    gesture_bin_plot.setAspectLocked(lock=True)
    # 绘制位点、参考线的窗口
    gesture_pos_view = sub_win5.addPlot()
    gesture_pos_view.invertY(True)  # 反转Y轴
    gesture_pos_view.setMouseEnabled(x=False, y=False)
    gesture_pos_view.setXRange(1, optical_size[0])
    gesture_pos_view.setYRange(1, optical_size[1])
    gesture_pos_view.showAxis('left', show=True)
    gesture_pos_view.showAxis('bottom', show=True)
    gesture_pos_view.showAxis('right', show=True)
    gesture_pos_view.showAxis('top', show=True)
    gesture_pos_view.setAspectLocked(lock=True)
    gesture_line_x = pg.InfiniteLine(movable=False, angle=90, label='x={value:0.2f}',
                                     pen=pg.mkPen(color=(200, 200, 100), width=5),
                                     labelOpts={'position': 0.3, 'color': (200, 200, 100),
                                                'fill': (200, 200, 200, 50), 'movable': True})
    gesture_line_y = pg.InfiniteLine(movable=False, angle=0, label='y={value:0.2f}',
                                     pen=pg.mkPen(color=(200, 200, 100), width=5),
                                     labelOpts={'position': 0.3, 'color': (200, 200, 100),
                                                'fill': (200, 200, 200, 50), 'movable': True})
    gesture_pos_view.addItem(gesture_line_x)
    gesture_pos_view.addItem(gesture_line_y)
    sub_win5.nextRow()
    # 绘制方向的窗口
    gesture_direction_view = sub_win5.addPlot(colspan=2)
    gesture_direction_view.invertY(True)  # 反转Y轴
    gesture_direction_view.setMouseEnabled(x=False, y=False)
    gesture_direction_view.setXRange(-1, 1)
    gesture_direction_view.setYRange(-1, 1)
    gesture_direction_view.showAxis('left', show=True)
    gesture_direction_view.showAxis('bottom', show=True)
    gesture_direction_view.showAxis('right', show=True)
    gesture_direction_view.showAxis('top', show=True)
    gesture_direction_view.setAspectLocked(lock=True)
    gesture_direction_graph = pg.GraphItem()
    gesture_pos = numpy.array([[0, 0], [0, 0]])  # 点的位置
    gesture_adj = numpy.array([[0, 1]])
    gesture_lines = numpy.array([(0, 0, 0, 255, 2)],
                                dtype=[('red', numpy.ubyte), ('green', numpy.ubyte),
                                ('blue', numpy.ubyte), ('alpha', numpy.ubyte), ('width', float)])
    gesture_symbols = ['o', '+']
    gesture_size = [0.2, 0.2]
    gesture_direction_graph.setData(pos=gesture_pos, adj=gesture_adj, pen=gesture_lines, size=gesture_size,
                                    symbol=gesture_symbols, pxMode=False)
    gesture_direction_view.addItem(gesture_direction_graph)
    # 添加绘图组件到dock中
    d5.addWidget(sub_win5)

    """Dock6 签字交互"""
    # 绘制二值图像的窗口
    sign_bin_plot = sub_win6.addPlot()
    sign_bin_plot.invertY(True)  # 反转Y轴
    sign_bin_plot.setMouseEnabled(x=False, y=False)
    sign_bin_plot.showAxis('left', show=False)
    sign_bin_plot.showAxis('bottom', show=False)
    sign_bin_display = pg.ImageItem()  # 二值图像
    sign_bin_plot.addItem(sign_bin_display)
    sign_bin_plot.setAspectLocked(lock=True)
    # 绘制位点、参考线的窗口
    sign_pos_view = sub_win6.addPlot()
    sign_pos_view.invertY(True)  # 反转Y轴
    sign_pos_view.setMouseEnabled(x=False, y=False)
    sign_pos_view.setXRange(1, optical_size[0])
    sign_pos_view.setYRange(1, optical_size[1])
    sign_pos_view.showAxis('left', show=True)
    sign_pos_view.showAxis('bottom', show=True)
    sign_pos_view.showAxis('right', show=True)
    sign_pos_view.showAxis('top', show=True)
    sign_pos_view.setAspectLocked(lock=True)
    sign_line_x = pg.InfiniteLine(movable=False, angle=90, label='x={value:0.2f}',
                                  pen=pg.mkPen(color=(200, 200, 100), width=5),
                                  labelOpts={'position': 0.3, 'color': (200, 200, 100),
                                             'fill': (200, 200, 200, 50), 'movable': True})
    sign_line_y = pg.InfiniteLine(movable=False, angle=0, label='y={value:0.2f}',
                                  pen=pg.mkPen(color=(200, 200, 100), width=5),
                                  labelOpts={'position': 0.3, 'color': (200, 200, 100),
                                             'fill': (200, 200, 200, 50), 'movable': True})
    sign_pos_view.addItem(sign_line_x)
    sign_pos_view.addItem(sign_line_y)
    sub_win6.nextRow()
    # 存储签字结果的窗口
    sign_accumulate_plot = sub_win6.addPlot(colspan=2)
    sign_accumulate_plot.invertY(True)  # 反转Y轴
    sign_accumulate_plot.setMouseEnabled(x=False, y=False)
    sign_accumulate_plot.showAxis('left', show=False)
    sign_accumulate_plot.showAxis('bottom', show=False)
    sign_accumulate_display = pg.ImageItem()  # 匹配结果
    sign_accumulate_plot.addItem(sign_accumulate_display)
    sign_accumulate_plot.setAspectLocked(lock=True)
    # 添加绘图组件到dock中
    d6.addWidget(sub_win6)

    """Dock7 状态机控制面板"""
    state_panel = pg.LayoutWidget()
    state_label = QtWidgets.QLabel("Current States:")  # 状态提示
    saveBtn = QtWidgets.QPushButton('Save Data')  # 存储指纹按钮
    goScene0Btn = QtWidgets.QPushButton('GO[Scene-0]')  # 跳转到[场景0]状态
    goScene1Btn = QtWidgets.QPushButton('GO[Scene-1]')  # 跳转到[场景1]状态
    goScene2Btn = QtWidgets.QPushButton('GO[Scene-2]')  # 跳转到[场景2]状态
    goScene3Btn = QtWidgets.QPushButton('GO[Scene-3]')  # 跳转到[场景3]状态
    goIDLE_CaptureBtn = QtWidgets.QPushButton('GO[IDLE_Capture]')  # 跳转到[IDLE_Capture]状态
    goFinger_HMIBtn = QtWidgets.QPushButton('GO[Finger_HMI]')  # 跳转到[Finger_HMI]状态
    goIDLE_SweepBtn = QtWidgets.QPushButton('GO[IDLE_Sweep]')  # 跳转到[IDLE_Sweep]状态
    goSweep_HMIBtn = QtWidgets.QPushButton('GO[Sweep_HMI]')  # 跳转到[Sweep_HMI]状态
    goIDLE_TapBtn = QtWidgets.QPushButton('GO[IDLE_Tap]')  # 跳转到[IDLE_Tap]状态
    goTap_HMIBtn = QtWidgets.QPushButton('GO[Tap_HMI]')  # 跳转到[Tap_HMI]状态
    goIDLE_GestureBtn = QtWidgets.QPushButton('GO[IDLE_Gesture]')  # 跳转到[IDLE_Gesture]状态
    goGesture_HMIBtn = QtWidgets.QPushButton('GO[Gesture_HMI]')  # 跳转到[Gesture_HMI]状态
    goIDLE_SignBtn = QtWidgets.QPushButton('GO[IDLE_Sign]')  # 跳转到[IDLE_Sign]状态
    goSign_HMIBtn = QtWidgets.QPushButton('GO[Sign_HMI]')  # 跳转到[Sign_HMI]状态
    # 组件布局
    state_panel.addWidget(state_label, row=0, col=0)
    state_panel.addWidget(saveBtn, row=0, col=1)
    state_panel.addWidget(goIDLE_CaptureBtn, row=1, col=0)
    state_panel.addWidget(goFinger_HMIBtn, row=1, col=1)
    state_panel.addWidget(goIDLE_SweepBtn, row=2, col=0)
    state_panel.addWidget(goSweep_HMIBtn, row=2, col=1)
    state_panel.addWidget(goIDLE_TapBtn, row=3, col=0)
    state_panel.addWidget(goTap_HMIBtn, row=3, col=1)
    state_panel.addWidget(goIDLE_GestureBtn, row=4, col=0)
    state_panel.addWidget(goGesture_HMIBtn, row=4, col=1)
    state_panel.addWidget(goIDLE_SignBtn, row=5, col=0)
    state_panel.addWidget(goSign_HMIBtn, row=5, col=1)
    state_panel.addWidget(goScene0Btn, row=6, col=0)
    state_panel.addWidget(goScene1Btn, row=6, col=1)
    state_panel.addWidget(goScene2Btn, row=7, col=0)
    state_panel.addWidget(goScene3Btn, row=7, col=1)
    d7.addWidget(state_panel)
    # 事件链接
    saveBtn.clicked.connect(capture_image)
    goIDLE_CaptureBtn.clicked.connect(goto_IDLE_Capture)
    goFinger_HMIBtn.clicked.connect(goto_Finger_HMI)
    goIDLE_SweepBtn.clicked.connect(goto_IDLE_Sweep)
    goSweep_HMIBtn.clicked.connect(goto_Sweep_HMI)
    goIDLE_TapBtn.clicked.connect(goto_IDLE_Tap)
    goTap_HMIBtn.clicked.connect(goto_Tap_HMI)
    goIDLE_GestureBtn.clicked.connect(goto_IDLE_Gesture)
    goGesture_HMIBtn.clicked.connect(goto_Gesture_HMI)
    goIDLE_SignBtn.clicked.connect(goto_IDLE_Sign)
    goSign_HMIBtn.clicked.connect(goto_Sign_HMI)
    goScene0Btn.clicked.connect(goto_scene_0)
    goScene1Btn.clicked.connect(goto_scene_1)
    goScene2Btn.clicked.connect(goto_scene_2)
    goScene3Btn.clicked.connect(goto_scene_3)

    """Dock8 电容屏读出系统控制面板"""
    cap_panel = pg.LayoutWidget()
    cap_param_apply = QtWidgets.QPushButton('Apply Param')  # 进行参数更新
    cap_reset = QtWidgets.QPushButton('Reset System')  # 系统重置
    cap_purge = QtWidgets.QPushButton('Clear Cache')  # 清除缓存
    save_cap_image = QtWidgets.QPushButton('Save Image')  # 保存图片

    label_cap_vbias3 = QtWidgets.QLabel("V Bias3")
    spin_cap_vbias3 = pg.SpinBox(value=v_bias3_cap, suffix='V', bounds=[-5, 5], finite=True, minStep=0.2, step=0.2)

    label_cap_vbias4 = QtWidgets.QLabel("V Bias4")
    spin_cap_vbias4 = pg.SpinBox(value=v_bias4_cap, suffix='V', bounds=[-5, 5], finite=True, minStep=0.2, step=0.2)

    label_cap_von = QtWidgets.QLabel("V On")
    spin_cap_von = pg.SpinBox(value=v_on_cap, suffix='V', bounds=[-15, 15], finite=True, minStep=0.2, step=0.2)

    label_cap_voff = QtWidgets.QLabel("V Off")
    spin_cap_voff = pg.SpinBox(value=v_off_cap, suffix='V', bounds=[-15, 15], finite=True, minStep=0.2, step=0.2)

    # 组件布局
    cap_panel.addWidget(label_cap_vbias3, row=0, col=0)
    cap_panel.addWidget(spin_cap_vbias3, row=0, col=1)
    cap_panel.addWidget(label_cap_vbias4, row=1, col=0)
    cap_panel.addWidget(spin_cap_vbias4, row=1, col=1)
    cap_panel.addWidget(label_cap_von, row=2, col=0)
    cap_panel.addWidget(spin_cap_von, row=2, col=1)
    cap_panel.addWidget(label_cap_voff, row=3, col=0)
    cap_panel.addWidget(spin_cap_voff, row=3, col=1)
    cap_panel.addWidget(cap_param_apply, row=4, col=0, colspan=2)
    cap_panel.addWidget(cap_reset, row=5, col=0)
    cap_panel.addWidget(cap_purge, row=5, col=1)
    cap_panel.addWidget(save_cap_image, row=6, colspan=2)
    d8.addWidget(cap_panel)
    # 事件链接
    spin_cap_vbias3.sigValueChanged.connect(change_vbias3_cap)
    spin_cap_vbias4.sigValueChanged.connect(change_vbias4_cap)
    spin_cap_von.sigValueChanged.connect(change_von_cap)
    spin_cap_voff.sigValueChanged.connect(change_voff_cap)
    cap_param_apply.clicked.connect(set_cap_param)
    cap_purge.clicked.connect(cap_cache_clear)
    cap_reset.clicked.connect(cap_sys_reset)
    save_cap_image.clicked.connect(save_image_cap)

    """Dock9 光读出系统控制面板"""
    optical_panel = pg.LayoutWidget()
    cali_apply = QtWidgets.QPushButton('Apply Cali')  # 应用标定按钮
    cali_bright = QtWidgets.QPushButton('Bright Cali')  # 标定亮场按钮
    cali_dark = QtWidgets.QPushButton('Dark Cali')  # 标定暗场按钮
    optical_param_apply = QtWidgets.QPushButton('Apply Param')  # 进行参数更新
    save_optical_image = QtWidgets.QPushButton('Save Image')  # 保存图片

    label_optical_vbias1 = QtWidgets.QLabel("V Bias1")
    spin_optical_vbias1 = pg.SpinBox(value=v_bias1_optical, suffix='V', bounds=[-5, 5], finite=True, minStep=0.2,
                                     step=0.2)

    label_optical_vbias2 = QtWidgets.QLabel("V Bias2")
    spin_optical_vbias2 = pg.SpinBox(value=v_bias2_optical, suffix='V', bounds=[-5, 5], finite=True, minStep=0.2,
                                     step=0.2)

    label_optical_von = QtWidgets.QLabel("V On")
    spin_optical_von = pg.SpinBox(value=v_on_optical, suffix='V', bounds=[-15, 15], finite=True, minStep=0.2, step=0.2)

    label_optical_voff = QtWidgets.QLabel("V Off")
    spin_optical_voff = pg.SpinBox(value=v_off_optical, suffix='V', bounds=[-15, 15], finite=True, minStep=0.2,
                                   step=0.2)

    label_optical_ilevel = QtWidgets.QLabel("I Level")
    spin_optical_ilevel = pg.SpinBox(value=i_level_optical, int=True, suffix='nA', bounds=[0, 200], finite=True,
                                     minStep=1, step=1)

    # 组件布局
    optical_panel.addWidget(label_optical_vbias1, row=0, col=0)
    optical_panel.addWidget(spin_optical_vbias1, row=0, col=1)
    optical_panel.addWidget(label_optical_vbias2, row=1, col=0)
    optical_panel.addWidget(spin_optical_vbias2, row=1, col=1)
    optical_panel.addWidget(label_optical_von, row=2, col=0)
    optical_panel.addWidget(spin_optical_von, row=2, col=1)
    optical_panel.addWidget(label_optical_voff, row=3, col=0)
    optical_panel.addWidget(spin_optical_voff, row=3, col=1)
    optical_panel.addWidget(label_optical_ilevel, row=4, col=0)
    optical_panel.addWidget(spin_optical_ilevel, row=4, col=1)
    optical_panel.addWidget(optical_param_apply, row=5, col=0, colspan=2)
    optical_panel.addWidget(cali_bright, row=6, col=0)
    optical_panel.addWidget(cali_dark, row=6, col=1)
    optical_panel.addWidget(cali_apply, row=7, col=0, colspan=2)
    optical_panel.addWidget(save_optical_image, row=8, colspan=2)
    d9.addWidget(optical_panel)
    # 事件链接
    spin_optical_vbias1.sigValueChanged.connect(change_vbias1_optical)
    spin_optical_vbias2.sigValueChanged.connect(change_vbias2_optical)
    spin_optical_von.sigValueChanged.connect(change_von_optical)
    spin_optical_voff.sigValueChanged.connect(change_voff_optical)
    spin_optical_ilevel.sigValueChanged.connect(change_ilevel_optical)
    optical_param_apply.clicked.connect(set_optical_param)
    cali_bright.clicked.connect(get_bright_data)
    cali_dark.clicked.connect(get_dark_data)
    cali_apply.clicked.connect(get_cali_param)
    save_optical_image.clicked.connect(save_image_optical)

    """状态检查和跳转Timer"""
    state_timer = QtCore.QTimer()
    state_timer.timeout.connect(state_transfer)
    state_timer.start(1)

    """单次触发的Timer"""
    single_shot_timer1 = QtCore.QTimer()
    single_shot_timer1.setSingleShot(True)  # 只启动一次
    single_shot_timer1.timeout.connect(goto_IDLE_Finger)  # 用于等待捕捉

    single_shot_timer2 = QtCore.QTimer()
    single_shot_timer2.setSingleShot(True)  # 只启动一次
    single_shot_timer2.timeout.connect(goto_IDLE_Sign)  # 用于跳转到签字

    """启动GUI"""
    win.show()
    app.exec()
