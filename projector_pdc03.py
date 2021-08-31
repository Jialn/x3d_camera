# Copyright (c) 2020. All Rights Reserved.
# Created on 2021-09-05
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
A low level python interface to control PDC03 projector
Test: python -m projector_pdc03
"""
import serial
import time

MAIN_CMD1 = int('0x41', 16)
MAIN_CMD2 = int('0x44', 16)

CMD_SHAKE_HANDS_OLD = int('0x4105', 16)
CMD_SHAKE_HANDS     = int('0x4400', 16)
CMD_SCAN_ONE        = int('0x4404', 16)
CMD_SCAN_ALL        = int('0x4401', 16)
CMD_SCAN_ANY        = int('0x4415', 16)

use_real_serial = True  # use real serial or only for test mode


class PyPDC():
    """ Class for PDC03 projector
    """

    def __init__(self, port=None, logging=False):
        """
        Init

        Args:
            port (string): the serial port, e.g, "COM3"
            logging (bool): log or not.
        """
        self._logging = logging
        if use_real_serial:
            self.ser = serial.Serial(port, 9600, timeout=2)
            self.is_open = self.ser.isOpen()
            if self.is_open:
                if self._logging: print('serial: %s open' % self.ser.name)
            else:
                if self._logging: print('failed to open serial port')

    def check_sum(self, data, cnt):
        res = 0
        for i in range(cnt):
            if i < len(data):
                res += data[i]
        if res & 0xff:
            if self._logging: print("checksum error! :" + str(res))
            return False
        return True

    def check_sum_gen(self, data, offset):
        res = 0
        count = data[offset] - 1
        for i in range(count):
            res += data[i]
        res = ~(res & 255)
        res += 1
        return res & 255

    def send_package(self, cmd, send_buf, TransmitSize, ReceiveSize, ReceiveTryCnt):
        main_cmd = (cmd >> 8) & 255
        SubCmd = cmd & 255

        if main_cmd == MAIN_CMD2:
            TotalSize = TransmitSize + 5
        else:
            TotalSize = TransmitSize + 4

        buffer = [int('0xe0', 16), TotalSize, main_cmd, SubCmd]
        for byte in send_buf:
            buffer.append(byte)

        if main_cmd == MAIN_CMD2:
            chk_sum = self.check_sum_gen(buffer, 1)
            buffer.append(chk_sum)

        buf_bytes = bytes(buffer[:TotalSize])
        if self._logging:
            if self._logging: print("to send:")
            if self._logging: print(buffer[:TotalSize])
        if use_real_serial:
            self.ser.write(buf_bytes)
            # time.sleep(0.01)

        if ReceiveSize == 0:
            return 0

        if main_cmd == MAIN_CMD2:
            TotalSize = ReceiveSize + 2
        else:
            TotalSize = ReceiveSize

        for i in range(ReceiveTryCnt):
            if use_real_serial:
                buffer = self.ser.read(TotalSize)
                if self._logging:
                    if self._logging: print("read")
                    if self._logging: print(buffer)
            else:
                # for test
                buffer = [ord('a')] * (TotalSize)
                buffer = bytes(buffer)

            buffer = list(buffer)
            if self._logging:
                if self._logging: print("recv:")
                if self._logging: print(buffer)
            if main_cmd == MAIN_CMD1 or self.check_sum(buffer, TotalSize) == True:
                ReceiveBuffer = buffer[1:ReceiveSize + 1]
                return ReceiveBuffer
        return 0

    def shake_hands(self):
        """
        Shake hands
        """
        spac_char = "ShakeHands"
        spac_bytes = spac_char.encode()
        spac_lists = list(spac_bytes)
        if self._logging: print(spac_lists)
        buffer = self.send_package(CMD_SHAKE_HANDS, spac_lists, 10, 10, 0)
        if self._logging: print(buffer)
        if buffer:
            if buffer == spac_lists:
                return True
            else:
                return False
        return False

    def scan_all_pattern(self, interval_time, start_index, display_time):
        buffer = [0] * 6
        buffer[0] = (interval_time & 255)
        buffer[1] = ((interval_time >> 8) & 255)
        buffer[2] = (start_index & 255)
        buffer[3] = ((start_index >> 8) & 255)
        buffer[4] = (display_time & 255)
        buffer[5] = ((display_time >> 8) & 255)
        res = self.send_package(CMD_SCAN_ALL, buffer, 6, 3, 0)
        if res == 0:
            if self._logging: print("ERROR, scan pattern send send_package return 0")
            return 0
        if res[0] == ord('A') and res[1] == ord('c') and res[2] == ord('k'):
            return True
        else:
            if self._logging: print("scan pattern return false")
            return False

    def scan_one_pattern(self, interval_time, index, display_time):
        buffer    = [0] * 6
        buffer[0] = (index & 255)
        buffer[1] = ((index >> 8) & 255)
        buffer[2] = (interval_time & 255)
        buffer[3] = ((interval_time >> 8) & 255)
        buffer[4] = (display_time & 255)
        buffer[5] = ((display_time >> 8) & 255)
        res = self.send_package(CMD_SCAN_ONE, buffer, 6, 3, 0)
        if res == 0:
            if self._logging: print("ERROR, scan pattern send send_package return 0")
            return 0
        if res[0] == ord('A') and res[1] == ord('c') and res[2] == ord('k'):
            return True
        else:
            if self._logging: print("scan pattern return false")
            return False

    def turn_led(self, on_off):
        # on_off : 0 or 1
        self.send_package(int('0x441B', 16), [on_off], 1, 3, 1)

    def close(self):
        if use_real_serial:
            self.ser.close()


# A simple test
if __name__ == "__main__":
    pdc = PyPDC(port="/dev/ttyUSB0", logging=True)
    if pdc.shake_hands():
        print("shake_hands suc")
    else:
        print("shake_hands err!")
    pdc.scan_one_pattern(interval_time=0, index=0, display_time=1000)
    time.sleep(2)
    pdc.scan_one_pattern(interval_time=0, index=1, display_time=1000)
    time.sleep(2)
    pdc.scan_one_pattern(interval_time=0, index=18, display_time=1000)
    time.sleep(2)
    pdc.scan_all_pattern(40, 0, 20)
    time.sleep(2)
    pdc.scan_all_pattern(40, 0, 20)
    time.sleep(2)
    pdc.turn_led(0)
    pdc.close()
