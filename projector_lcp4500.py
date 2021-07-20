
import time
import hid
from config import Config

"""
Test: python -m projector_lcp4500

Note: To use python hid, please make sure below libs are installed in linux:
$ pip install hidapi

If met error, try the following:
$ sudo apt-get install python-dev libusb-1.0-0-dev libudev-dev
$ pip install --upgrade setuptools

DLPC350 Commands are combine with CMD2 and CMD3. Therefore bytes in each Cmdlist are
CMD2 CMD3 and bytes appropriate to be send.
"""

USB_MAX_PACKET_SIZE = 64
# DLP4500's USB Vid and Pid is 0x0451 and 0x6401 separately.
Vid = 0x0451
Pid = 0x6401
insert_delay = 0.003

Cmdlist = {
    'STATUS_HW': [0x1A, 0x0A, 0x00],
    'STATUS_SW': [0x1A, 0x0B, 0x00],
    'LED_ENABLE': [0x1A, 0x07, 0x01],
    'STATUS_SYS': [0x1A, 0x0B, 0x00],
    'STATUS_MAIN': [0x1A, 0x0C, 0x00],
    'GET_VERSION': [0x02, 0x05, 0x00],
    'VID_SIG_STAT': [0x07, 0x1C, 0x1C],
    'SOURCE_SEL': [0x1A, 0x00, 0x01],
    'PIXEL_FORMAT': [0x1A, 0x02, 0x01],
    'CHANNEL_SWAP': [0x1A, 0x37, 0x01],
    'POWER_CONTROL': [0x02, 0x00, 0x01],
    'FLIP_LONG': [0x10, 0x08, 0x01],
    'FLIP_SHORT': [0x10, 0x09, 0x01],
    'TPG_SEL': [0x12, 0x03, 0x01],
    'PWM_INVERT': [0x1A, 0x05, 0x01],
    'GET_FW_TAG_INFO': [0x1A, 0xFF, 0x00],
    'SW_RESET': [0x08, 0x02, 0x00],
    'DMD_PARK': [0x00, 0x00, 0x00],
    'BUFFER_FREEZE': [0x10, 0x0A, 0x01],
    'PWM_ENABLE': [0x1A, 0x10, 0x01],
    'PWM_SETUP': [0x1A, 0x11, 0x06],
    'PWM_CAPTURE_CONFIG': [0x1A, 0x12, 0x05],
    'GPIO_CTRL': [0x1A, 0x38, 0x02],
    'LED_CURRENT': [0x0B, 0x01, 0x03],
    'DISP_CONFIG': [0x10, 0x00, 0x10],
    'MEM_CTRL': [0x1A, 0x16, 0x09],
    'LUT_VALID': [0x1A, 0x1A, 0x01],
    'DISP_MODE': [0x1A, 0x1B, 0x01],
    'TRIG_OUT1_CTL': [0x1A, 0x1D, 0x03],
    'TRIG_OUT2_CTL': [0x1A, 0x1E, 0x02],
    'RED_STROBE_DLY': [0x1A, 0x1F, 0x02],
    'GRN_STROBE_DLY': [0x1A, 0x20, 0x02],
    'BLU_STROBE_DLY': [0x1A, 0x21, 0x02],
    'PAT_DISP_MODE': [0x1A, 0x22, 0x01],
    'PAT_TRIG_MODE': [0x1A, 0x23, 0x01],
    'PAT_START_STOP': [0x1A, 0x24, 0x01],
    'PAT_EXPO_PRD': [0x1A, 0x29, 0x08],
    'INVERT_DATA': [0x1A, 0x30, 0x01],
    'PAT_CONFIG': [0x1A, 0x31, 0x04],
    'MBOX_ADDRESS': [0x1A, 0x32, 0x01],
    'MBOX_CTL': [0x1A, 0x33, 0x01],
    'MBOX_DATA': [0x1A, 0x34, 0x00],
    'TRIG_IN1_DELAY': [0x1A, 0x35, 0x04],
    'TRIG_IN2_DELAY': [0x1A, 0x36, 0x01],
    'IMAGE_LOAD': [0x1A, 0x39, 0x01],
    'IMAGE_LOAD_TIMING': [0x1A, 0x3A, 0x02],
    'MBOX_EXP_DATA': [0x1A, 0x3E, 0x0C],
    'MBOX_EXP_ADDRESS': [0x1A, 0x3F, 0x02],
    'EXP_PAT_CONFIG': [0x1A, 0x40, 0x06],
    'NUM_IMAGE_IN_FLASH': [0x1A, 0x42, 0x01],
    'TPG_COLOR': [0x12, 0x04, 0x0C],
    'PWM_CAPTURE_READ': [0x1A, 0x13, 0x05],
}
BIT0 = 0x01
BIT1 = 0x02
BIT2 = 0x04
BIT3 = 0x08
BIT4 = 0x10
BIT5 = 0x20
BIT6 = 0x40
BIT7 = 0x80
Pat_Num_MONO_1BPP = {
    'G0': 0,
    'G1': 1,
    'G2': 2,
    'G3': 3,
    'G4': 4,
    'G5': 5,
    'G6': 6,
    'G7': 7,
    'R0': 8,
    'R1': 9,
    'R2': 10,
    'R3': 11,
    'R4': 12,
    'R5': 13,
    'R6': 14,
    'R7': 15,
    'B0': 16,
    'B1': 17,
    'B2': 18,
    'B3': 19,
    'B4': 20,
    'B5': 21,
    'B6': 22,
    'B7': 23,
    'BLACK': 24
}
Pat_Num_MONO_8BPP = {
    'G7_to_G0': 0,
    'R7_to_R0': 1,
    'B7_to_B0': 2
}
MINIMUM_EXPOSURE_TIME = {
    'MONO_1BPP': 235,
    'MONO_2BPP': 700,
    'MONO_3BPP': 1570,
    'MONO_4BPP': 1700,
    'MONO_5BPP': 2000,
    'MONO_6BPP': 2500,
    'MONO_7BPP': 4500,
    'MONO_8BPP': 8333
}

Set_Debug_info = False


class Dlp_usb:
    """
    Class for hid usb communication.
    """

    def __init__(self):
        self.isUSBConnected = False
        self.dev = hid.device()

    def dlp_usb_open(self):
        try:
            self.dev.open(Vid, Pid)
            self.dev.set_nonblocking(1)  # Enable non-blocking mode
        except OSError as oe:
            print(oe)
            print("Projector open failed. Check connection!")
        else:
            self.isUSBConnected = True

    def dlp_usb_close(self):
        try:
            self.dev.close()
        except OSError as oe:
            print(oe)
            print("Projector close failed!")
        else:
            self.isUSBConnected = False

    def dlp_usb_write(self, send_data):
        """
        :param send_data: Data has to be a list contains integers(0-255) and length smaller than 65.
        """
        if len(send_data) > 65:
            print('USB Data package is larger than 65 bytes. ')
            return -2
        bytes_written = self.dev.write(send_data)
        if bytes_written == -1:
            print("USB Send error!")
            self.dlp_usb_close()
            self.isUSBConnected = False
            return -1
        time.sleep(insert_delay)
        return bytes_written

    def dlp_usb_read(self, max_length):
        """
        :return: Return a list contains integers(0-255) up to max_length bytes
        """
        read_data = self.dev.read(max_length)
        time.sleep(insert_delay)
        return read_data

class Dlp350_api:
    """
    Class for dlpc350 controller command.
    """

    def __init__(self):
        self.usb = Dlp_usb()
        self.seqNum = 0
        self.g_PatLut = []
        self.g_ExpLut = []
        self.pat_valid = False

    def dlp350_connect(self):
        self.usb.dlp_usb_open()
        if self.usb.isUSBConnected:
            return 0
        else:
            return -1

    def dlp350_disconnect(self):
        self.usb.dlp_usb_close()
        if not self.usb.isUSBConnected:
            return 0
        else:
            return -1

    def dlp350_continueRead(self):
        return self.usb.dlp_usb_read(64)

    def dlp350_SendMessage(self, msg_pkg=[], ackRequired=True):
        """
        :param msg_pkg: list contains bytes send to device
        :param ackRequired: if need read from device,default is True
        :return:
        A list: that from device when ackRequired is set;
        -3: if read nothing from device;
        -2: if nonack from device;
        -1: Write failed
        bytes_send: Bytes number that been written to device
        """
        bytes_send = 0
        for msg in msg_pkg:
            ret = self.usb.dlp_usb_write(msg)
            if ret == -1:
                return -1
            else:
                bytes_send = bytes_send + ret
        if ackRequired:
            time.sleep(
                insert_delay)  # If not delay seems to be can't read data from device. Needs to figure out minimum delay
            read_data = self.usb.dlp_usb_read(64)
            if len(read_data) == 0:
                if Set_Debug_info:
                    print('USB Read nothing from device.')
                return -3
            else:
                if read_data[0] & 0x20 > 0:
                    return -2
                else:
                    return read_data
        else:
            return bytes_send

    def dlp350_PrepReadCMD(self, CMD=Cmdlist['STATUS_HW'], param=None):
        msg = HidMessageStruct(CMD)
        msg.reply = 1
        if param is None:
            msg.data_len = 2
        else:
            msg.data_len = 2 + len(param)
            msg.cmddata[0:len(param)] = param
        return msg.Message_to_Data()

    def dlp350_PrepWriteCMD(self, CMD=Cmdlist['LED_ENABLE'], data=[0], ackRequired=True):
        msg = HidMessageStruct(CMD, ackRequired)
        msg.rw = 0
        self.seqNum = 0
        msg.seq = self.seqNum
        msg.cmddata[0:len(data)] = data
        return msg.Message_to_Data()

    def dlp350_ReadFwTag(self):
        """

        :return: str=Firmware tag
                 -1=Failed
        """
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['GET_FW_TAG_INFO'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            i = 4
            fw_tag_info = ''
            while (readbytes[i] != 0) and (i < 36):
                fw_tag_info = fw_tag_info + chr(readbytes[i])
                i += 1
            return fw_tag_info
        else:
            print('Dlp350 Read Firwmare tag failed.')
            return -1

    def dlp350_GetLedEnables(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['LED_ENABLE'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if readbytes[4] & BIT0:
                red_led_enable = True
                if Set_Debug_info:
                    print('Red LED is enabled')
            else:
                red_led_enable = False
                if Set_Debug_info:
                    print('Red LED is disable')
            if readbytes[4] & BIT1:
                grn_led_enable = True
                if Set_Debug_info:
                    print('Green LED is enabled')
            else:
                grn_led_enable = False
                if Set_Debug_info:
                    print('Green LED is disable')
            if readbytes[4] & BIT2:
                blu_led_enable = True
                if Set_Debug_info:
                    print('Blue LED is enabled')
            else:
                blu_led_enable = False
                if Set_Debug_info:
                    print('Blue LED is disable')
            if readbytes[4] & BIT3:
                all_led_enable = True
                if Set_Debug_info:
                    print('All LED is enabled')
            else:
                all_led_enable = False
                if Set_Debug_info:
                    print('All LED is disable')
            return grn_led_enable, grn_led_enable, blu_led_enable, all_led_enable
        else:
            return -1

    def dlp350_SetLedEnables(self, all_led=True, red=False, grn=False, blu=False):
        """

        :param all_led:
        :param red:
        :param grn:
        :param blu:
        :return:
        """
        enable_byte = 0
        if all_led:
            enable_byte |= BIT3
        if red:
            enable_byte |= BIT0
        if grn:
            enable_byte |= BIT1
        if blu:
            enable_byte |= BIT2
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['LED_ENABLE'], data=[enable_byte], ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        return -1

    def dlp350_GetLedCurrents(self):
        """
        CMD2 0x0B, CMD3 0x01. Control the pulse duration of LED PWM modulation output pin. PWM value is set from 0 to 100% in 256 steps.
        If the LED PWM polarity is set to Normal polarity, a seting of 0xFF gives the maximum PWM current. Careful with LED current setting
        as it might harm the whole system.
        Reset value are:
                Red:   0x97
                Green: 0x78
                Blue:  0x7D
        :return: red_current_byte, grn_current_byte, blu_current_byte
                -1 if error occur
        """
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['LED_CURRENT'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            red_current_byte = readbytes[4]
            grn_current_byte = readbytes[5]
            blu_current_byte = readbytes[6]
            if Set_Debug_info:
                print("LED Current Setting are: RED: %d, GREEN: %d, BLUE: %d" % (
                    red_current_byte, grn_current_byte, blu_current_byte))
            return red_current_byte, grn_current_byte, blu_current_byte
        else:
            if Set_Debug_info:
                print('Get LED Current Setting failed.')
            return -1

    def dlp350_SetLedCurrent(self, red_current=0x97, grn_current=0x78, blu_current=0x7D):
        """

        :param red_current:
        :param grn_current:
        :param blu_current:
        :return:
        """
        if red_current > 255 or red_current < 0:
            print('Invalid Red LED Current setting. It has to be 0~255.')
            return -1
        if grn_current > 255 or grn_current < 0:
            print('Invalid Green LED Current setting. It has to be 0~255.')
            return -1
        if blu_current > 255 or blu_current < 0:
            print('Invalid Blue LED Current setting. It has to be 0~255.')
            return -1
        cmd_data = [red_current, grn_current, blu_current]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['LED_CURRENT'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print("Set LED Current failed.")
        return -1

    def dlp350_SetBufferFreeze(self, Freeze=True):
        """
        USB:CMD2:0x10,CMD3:0x0A
        :param Freeze: Set True to disable buffer swap. Default value is True when reset.
        :return:
        """
        cmd_data = [int(Freeze)]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['BUFFER_FREEZE'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Set Buffer Freeze register failed.')
        return -1

    def dlp350_GetBufferFreezeStatus(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['BUFFER_FREEZE'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if readbytes[4] == 1:
                if Set_Debug_info:
                    print('Buffer Swapping is now disable. ')
            else:
                if Set_Debug_info:
                    print('Buffer Swapping is now Enable. ')
            return readbytes[4]
        print('Get BufferFreezeStatus failed')
        return -1

    def dlp350_GetStatus(self):
        """
        Get status of DLPC350. HW Status contains status of sequencer, DMD Controller and initialization.
        SYS Status contains DLPC350 internal memory tests status.
        Main Status contains DLPC350 sequencer, frame buffer, gamma correction status.
        :return:
        hw_status, sys_status, main_status when success.
        -1 if error occur
        """
        msg_pkg_hw = self.dlp350_PrepReadCMD(CMD=Cmdlist['STATUS_HW'])
        readbytes_hw = self.dlp350_SendMessage(msg_pkg_hw)
        if type(readbytes_hw) is list:
            hw_status = readbytes_hw[6]
        else:
            return -1
        msg_pkg_sys = self.dlp350_PrepReadCMD(CMD=Cmdlist['STATUS_SYS'])
        readbytes_sys = self.dlp350_SendMessage(msg_pkg_sys)
        if type(readbytes_sys) is list:
            sys_status = readbytes_sys[6]
        else:
            return -1
        msg_pkg_main = self.dlp350_PrepReadCMD(CMD=Cmdlist['STATUS_MAIN'])
        readbytes_main = self.dlp350_SendMessage(msg_pkg_main)
        if type(readbytes_main) is list:
            main_status = readbytes_main[6]
        else:
            return -1
        return hw_status, sys_status, main_status

    def dlp350_SoftwareReset(self):
        """
        Use this api to reset controller
        :return: 0 success
                -1 failed
        """
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['SW_RESET'], ackRequired=True)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 reset failed.')
        return -1

    def dlp350_SetMode(self, SLmode=True):
        """
        USB: CMD2:0x1A, CMD3:0x1B
        The Display Mode selection command enables the DLPC internal image processing functions for
        video mode or bypasses them for pattern display mode. USB CMD is 0x1A,0x1B. Default is video mode after reset.
        :param SLmode: TRUE = Pattern display mode.
                       False = Video streaming mode.
        :return: 0=PASS
                -1=FAIL
        """
        cmd_data = [int(SLmode)]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['DISP_MODE'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 Set Display mode failed. ')
        return -1

    def dlp350_GetMode(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['DISP_MODE'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if readbytes[4] == 1:
                if Set_Debug_info:
                    print('Current Display mode is at Pattern display mode.')
                return 1
            else:
                if Set_Debug_info:
                    print('Current Display mode is at Video display mode. ')
            return 0
        else:
            print('Dlp350 Get Display mode failed.')
            return -1

    def dlp350_SetPowerMode(self, standby=False):
        """
        The Power Control places the DLPC350 in a low-power state and powers down the DMD interface.
        Standby mode should only be enabled after all data for the last frame to be displayed has been
        transferred to the DLPC350. Standby mode must be disabled prior to sending any new data. USB CMD is 0x02, 0x00
        Default is in Normal mode after reset.
        :param standby:TRUE = Standby mode. Places DLPC350 in low power state and powers down the DMD interface
        FALSE = Normal Mode.
        :return: 0 success
                -1 failed
        """
        cmd_data = [int(standby)]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['POWER_CONTROL'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 Set power Mode failed.')
        return -1

    def dlp350_GetPowerMode(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['POWER_CONTROL'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if readbytes[4] == 0:
                if Set_Debug_info:
                    print('Device works in normal operation. ')
                return 0
            else:
                if Set_Debug_info:
                    print('Device works in standby mode. ')
                return 1
        else:
            print('Dlp350 read Power Mode failed.')
            return -1

    def dlp350_SetInputSource(self, source=2, port_width=1):
        """
        The Input Source Selection command selects the input source to be displayed by the DLPC350: 30-bit
        Parallel Port, Internal Test Pattern, Flash memory, or FPD-link interface.
        :param source:Select the input source and interface mode:
                0 = Parallel interface with 8-bit, 16-bit, 20-bit, 24-bit, or 30-bit RGB or YCrCb data formats
                1 = Internal test pattern; Use DLPC350_SetTPGSelect() API to select pattern
                2 = Flash. Images are 24-bit single-frame, still images stored in flash that are uploaded on command.
                3 = FPD-link interface
        :param port_width: - I - Parallel Interface bit depth
                0 = 30-bits
                1 = 24-bits
                2 = 20-bits
                3 = 16-bits
                4 = 10-bits
                5 = 8-bits
        :return:
        """
        cmd_data = [source + (port_width << 3)]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['SOURCE_SEL'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 Set Input source failed.')
        return -1

    def dlp350_GetInputSource(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['SOURCE_SEL'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            port_width = (readbytes[4] >> 3)
            source = (readbytes[4] & 0x7)
            if Set_Debug_info:
                print("Input source is %#x, Parallel port width is %#x" % (source, port_width))
            return source, port_width
        print('Get Input source setting failed. ')
        return -1

    def dlp350_SetPatternDisplayMode(self, external=False):
        """
        USB:CMD2:0x1A,CMD3:0x22
        The Pattern Display Data Input Source command selects the source of the data for pattern display:
        streaming through the 24-bit RGB/FPD-link interface or stored data in the flash image memory area from
        external Flash. Before executing this command, stop the current pattern sequence. After executing this
        command, send the Validation command (I2C: 0x7D or USB: 0x1A1A) once before starting the pattern sequence.
        :param external:TRUE = Pattern Display Data is streamed through the 24-bit RGB/FPD-link interface
                        FALSE = Pattern Display Data is fetched from flash memory
        :return:
        """
        if external:
            cmd_data = [0]
        else:
            cmd_data = [3]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['PAT_DISP_MODE'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Set pattern display mode failed.')
        return -1

    def dlp350_GetPatternDisplayMode(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['PAT_DISP_MODE'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if readbytes[4] == 3:
                if Set_Debug_info:
                    print('Pattern Display data source is from flash memory.')
                return 3
            else:
                if Set_Debug_info:
                    print('Pattern Display data stream through RGB/FPD Link.')
                return 0
        print('Get Pattern display mode failed.')
        return -1

    def dlp350_SetPixelFormat(self, format=0):
        """
        (USB: CMD2: 0x1A, CMD3: 0x02)
        This API defines the pixel data format input into the DLPC350.Refer to programmer's guide for supported pixel formats
        for each source type.
        :param format:Select the pixel data format:
            0 = RGB 4:4:4 (30-bit)
            1 = YCrCb 4:4:4 (30-bit)
            2 = YCrCb 4:2:2
        :return:
        """
        cmd_data = [format]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['PIXEL_FORMAT'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 Set Pixel Format failed. ')
        return -1

    def dlp350_GetPixelFormat(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['PIXEL_FORMAT'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if Set_Debug_info:
                print("Input Pixel format setting byte is %#x." % readbytes[4])
            return readbytes[4]
        else:
            return -1

    def dlp350_SetTPGSelect(self, pattern=8):
        """
        (USB: CMD2: 0x12, CMD3: 0x03)
        When the internal test pattern is the selected input, the Internal Test Patterns Select defines the test
        pattern displayed on the screen. These test patterns are internally generated and injected into the
        beginning of the DLPC350 image processing path. Therefore, all image processing is performed on the
        test images. All command registers should be set up as if the test images are input from an RGB 8:8:8
        external source.
        :param pattern: Selects the internal test pattern:
                0x0 = Solid Field
                0x1 = Horizontal Ramp
                0x2 = Vertical Ramp
                0x3 = Horizontal Lines
                0x4 = Diagonal Lines
                0x5 = Vertical Lines
                0x6 = Grid
                0x7 = Checkerboard
                0x8 = RGB Ramp
                0x9 = Color Bars
                0xA = Step Bars
        :return:
        """
        cmd_data = [pattern]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['TPG_SEL'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 set internal test pattern failed.')
        return -1

    def dlp350_GetTPGSelect(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['TPG_SEL'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if Set_Debug_info:
                print("Internal Test Patterns is selected as %#x" % readbytes[4])
            return readbytes[4]
        print('Dlp350 Get internal test pattern selection setting failed.')
        return -1

    def dlp350_LoadImageIndex(self, index=0):
        """
         (USB: CMD2: 0x1A, CMD3: 0x39)
        This command loads an image from flash memory and then performs a buffer swap to display the loaded
        image on the DMD.
        :param index: Image Index. Loads the image at this index from flash.
        :return:
        """
        cmd_data = [index]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['IMAGE_LOAD'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 load image from flash to internal buffer failed. ')
        return -1

    def dlp350_GetImageIndex(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['IMAGE_LOAD'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if Set_Debug_info:
                print("The most recent load image index is %d" % readbytes[4])
            return readbytes[4]
        print('Dlp350 get most recent load image index failed.')
        return -1

    def dlp350_GetNumImagesInFlash(self):
        """
        (USB: CMD2: 0x1A, CMD3: 0x42)
        This command reads number of images in the firmware running in DLPC350 controller.
        :return:
        """
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['NUM_IMAGE_IN_FLASH'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if Set_Debug_info:
                print("%d images in the flash." % readbytes[4])
            return readbytes[4]
        print('Dlp350 Get numbers of images in flash failed.')
        return -1

    def dlp350_ClearPatLut(self):
        self.g_PatLut = []

    def dlp350_ClearExpLut(self):
        self.g_ExpLut = []

    def dlp350_AddToPatLut(self, TrigType=0, PatNum=0, BitDepth=1, LEDSelect=7, InvertPat=False, InsertBlack=True,
                           BufSwap=False, trigOutPrev=False):
        lutWord = TrigType & 0x03
        if PatNum > 24:
            return -1
        lutWord |= ((PatNum & 0x3F) << 2)
        if (BitDepth > 8) or (BitDepth <= 0):
            return -1
        lutWord |= ((BitDepth & 0xF) << 8)
        if LEDSelect > 7:
            return -1
        lutWord |= ((LEDSelect & 0x7) << 12)
        if InvertPat:
            lutWord |= 0x10000
        if InsertBlack:
            lutWord |= 0x20000
        if BufSwap:
            lutWord |= 0x40000
        if trigOutPrev:
            lutWord |= 0x80000
        self.g_PatLut.append(lutWord)
        return 0

    def dlp350_AddToExpLut(self, TrigType=0, PatNum=0, BitDepth=1, LEDSelect=7, InvertPat=False, InsertBlack=True,
                           BufSwap=False,
                           trigOutPrev=False, exp_time_us=10000, ptn_frame_period_us=10000):
        lutWord = TrigType & 0x03
        if PatNum > 24:
            return -1
        lutWord |= ((PatNum & 0x3F) << 2)
        if (BitDepth > 8) or (BitDepth <= 0):
            return -1
        lutWord |= ((BitDepth & 0xF) << 8)
        if LEDSelect > 7:
            return -1
        lutWord |= ((LEDSelect & 0x7) << 12)
        if InvertPat:
            lutWord |= 0x10000
        if InsertBlack:
            lutWord |= 0x20000
        if BufSwap:
            lutWord |= 0x40000
        if trigOutPrev:
            lutWord |= 0x80000
        self.g_ExpLut.append([lutWord, exp_time_us, ptn_frame_period_us])
        return 0

    def dlp350_OpenMailBox(self, MboxNum):
        """
        (USB: CMD2: 0x1A, CMD3: 0x33)
        This API opens the specified Mailbox within the DLPC350 controller. This API must be called
        before sending data to the mailbox/LUT using DLPC350_SendPatLut() or DLPC350_SendImageLut() APIs.
        :param MboxNum:1 = Open the mailbox for image index configuration
                       2 = Open the mailbox for pattern definition.
                       3 = Open the mailbox for the Variable Exposure
        :return: 0 = Pass
                -1 = Failed
        """
        cmd_data = [MboxNum]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['MBOX_CTL'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            # print('**Open Mail Box action %d **' % MboxNum)
            return 0
        print('Open Mailbox action %d error.' % MboxNum)
        return -1

    def dlp350_CloseMailBox(self):
        """
        Close mailbox when written on lut complete.
        :return:
        """
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['MBOX_CTL'], data=[0], ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            # print('** Close Mailbox **')
            return 0
        print('Close mail box error')
        return -1

    def dlp350_SetMBoxAddr(self, Addr=0):
        """
        (USB: CMD2: 0x1A, CMD3: 0x32)
        This API defines the offset location within the DLPC350 mailboxes to write data into or to read data from
        :param Addr: 0-127 - Defines the offset within the selected (opened) LUT to write/read data to/from.
        :return:
        """
        if (Addr > 127) or (Addr < 0):
            print('MBox Addr has to be within 0-127.')
            return -1
        cmd_data = [Addr]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['MBOX_ADDRESS'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Set DLP350 Mail box addr %d error.' % Addr)
        return -1

    def dlp350_SetVarExpMboxAddr(self, Addr=0):
        """
        USB:CMD2:0x1A,CMD3:0x3F
        This API defines the offset location within the DLPC350 mailboxes to write data into or to read data from
        :param Addr: 0-1823 - Defines the offset within the selected (opened) LUT to write/read data to/from.
        :return:
        """
        if (Addr > 1823) or (Addr < 0):
            print('MBox VarExp Addr has to be within 0~1823.')
            return -1
        cmd_data = [Addr & 0xFF, Addr >> 8]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['MBOX_EXP_ADDRESS'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Set DLP350 VarExp Mail box addr %d error.' % Addr)
        return -1

    def dlp350_SendVarExpPatLut(self):
        """
        (USB: CMD2: 0x1A, CMD3: 0x3E)
        :return:
        """
        # Open mailbox for variable exposure pattern definition.
        if self.dlp350_OpenMailBox(3) < 0:
            return -1
        if len(self.g_ExpLut) == 0:
            print('ExpLut entries list is empty.')
            return -1
        for pat_index in range(len(self.g_ExpLut)):
            cmd_data = []
            if self.dlp350_SetVarExpMboxAddr(Addr=pat_index) < 0:
                return -1
            for word in self.g_ExpLut[pat_index]:
                cmd_data = cmd_data + [word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF, (word >> 24) & 0xFF]
            msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['MBOX_EXP_DATA'], data=cmd_data, ackRequired=False)
            if self.dlp350_SendMessage(msg_pkg, ackRequired=False) < 0:
                print('Error when sending %d var exp pattern Lut.' % pat_index)
                self.dlp350_CloseMailBox()
                return -1
        if self.dlp350_CloseMailBox() < 0:
            return -1
        return 0

    def dlp350_SendVarExpImageLut(self, LutEntries, numEntries):
        """
        This API sends the image LUT to the DLPC350 Controller
        :param LutEntries: A list contain LUT entries that to be sent.
        :param numEntries: number of entries to be sent to the controller
        :return:
        """
        if (numEntries < 1) or (numEntries > 256) or (len(LutEntries) == 0):
            return -1
        # Open Mailbox for image index configuration
        if self.dlp350_OpenMailBox(1) < 0:
            return -1
        # Set addr pointer to 0
        if self.dlp350_SetVarExpMboxAddr(0) < 0:
            return -1
        # Special Case if 2 entries
        if numEntries == 2:
            cmd_data = [LutEntries[1], LutEntries[0]]
        else:
            cmd_data = LutEntries
        CMD = Cmdlist['MBOX_DATA']
        CMD[2] = numEntries
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=CMD, data=cmd_data, ackRequired=False)
        ret = self.dlp350_SendMessage(msg_pkg, ackRequired=False)
        if self.dlp350_CloseMailBox() < 0:
            return -1
        if ret > 0:
            return 0
        print('Dlp350 Send varExp image Lut failed. ')
        return -1

    def dlp350_SendPatLut(self):
        """
        This API sends the pattern LUT created by calling dlp350_AddToPatLut() which store in class attribute g_PatLut
        Note: Before sending any mail box data, display mode, trigger mode, exposure, and frame rate has to be set in advance.
        :return:
        """
        # Set CMD as MBOX DATA USB 0x1A34, each Lut should contains 3 bytes
        CMD = Cmdlist['MBOX_DATA']
        CMD[2] = 3
        if len(self.g_PatLut) == 0:
            print("PatLut entries list is empty. ")
            return -1
        # Open Mailbox for pattern definition
        if self.dlp350_OpenMailBox(2) < 0:
            return -1
        # Send each Pat from Patlut list to its relevant address.
        for pat_index in range(len(self.g_PatLut)):
            if self.dlp350_SetMBoxAddr(pat_index) < 0:
                return -1
            cmd_data = [self.g_PatLut[pat_index] & 0xFF, (self.g_PatLut[pat_index] >> 8) & 0xFF,
                        (self.g_PatLut[pat_index] >> 16) & 0xFF]
            msg_pkg = self.dlp350_PrepWriteCMD(CMD=CMD, data=cmd_data, ackRequired=False)
            print('Patlut %d send:' % pat_index, end=' ')
            print(msg_pkg)
            if self.dlp350_SendMessage(msg_pkg, ackRequired=False) <= 0:
                self.dlp350_CloseMailBox()
                print('Error when sending %d pattern lut. ' % pat_index)
                return -1
        # Close Mailbox
        if self.dlp350_CloseMailBox() < 0:
            return -1
        return 0

    def dlp350_SendImageLut(self, LutEntries, numEntries):
        """
        This API sends the image LUT to the DLPC350 Controller
        :param LutEntries: LutEntries: A list contain LUT entries that to be sent.
        :param numEntries: number of entries to be sent to the controller
        :return:
        """
        if (numEntries < 1) or numEntries > 64 or (len(LutEntries) == 0):
            print("lut entries number exceed limit.")
            return -1
        # Open Mailbox for image index configuration
        if self.dlp350_OpenMailBox(1) < 0:
            return -1
        # Set pointer addr to 0
        if self.dlp350_SetMBoxAddr(0) < 0:
            return -1
        # Special case of 2 entries
        if numEntries == 2:
            cmd_data = [LutEntries[1], LutEntries[0]]
        else:
            cmd_data = LutEntries
        CMD = Cmdlist['MBOX_DATA']
        CMD[2] = numEntries
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=CMD, data=cmd_data, ackRequired=False)
        ret = self.dlp350_SendMessage(msg_pkg, ackRequired=False)
        # Close Mailbox
        if self.dlp350_CloseMailBox() < 0:
            return -1
        if ret > 0:
            return 0
        return -1

    def dlp350_GetPatLut(self, numEntries):
        """
        (USB: CMD2: 0x1A, CMD3: 0x34)
         This API reads the pattern LUT from the DLPC350 controller and stores it in the local list g_Patlut.
        :param numEntries:
        :return:
        """
        if numEntries > 128 or numEntries < 1:
            print('Number of entries is out of limit. ')
            return -1
        # Open Mailbox for pattern definition.
        if self.dlp350_OpenMailBox(2) < 0:
            return -1
        self.dlp350_ClearPatLut()
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['MBOX_DATA'])
        for lut_index in range(numEntries):
            if self.dlp350_SetMBoxAddr(lut_index) < 0:
                return -1
            readbytes = self.dlp350_SendMessage(msg_pkg)
            if type(readbytes) is list:
                self.g_PatLut.append(readbytes[4] | (readbytes[5] << 8) | (readbytes[6] << 16))
            else:
                self.dlp350_CloseMailBox()
                return -1
        if self.dlp350_CloseMailBox() < 0:
            return -1
        return 0

    def dlp350_GetVarExpPatLut(self, numEntries):
        """
        (USB: CMD2: 0x1A, CMD3: 0x3E)
        This API reads the pattern LUT from the DLPC350 controller and stores it in the local list g_Explut.
        :param numEntries: Number of entries expected in pattern LUT.
        :return:
        """
        if numEntries > 1824 or numEntries < 0:
            return -1
        # Open Mailbox for VarExp Pattern.
        if self.dlp350_OpenMailBox(3) < 0:
            return -1
        self.dlp350_ClearExpLut()
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['MBOX_EXP_DATA'])
        # Read lut data
        for lut_index in range(numEntries):
            if self.dlp350_SetVarExpMboxAddr(lut_index) < 0:
                return -1
            readbytes = self.dlp350_SendMessage(msg_pkg)
            if type(readbytes) is list:
                lut = readbytes[4] | (readbytes[5] << 8) | (readbytes[6] << 16) | (readbytes[7] << 24)
                exp_time_us = readbytes[8] | (readbytes[9] << 8) | (readbytes[10] << 16) | (readbytes[11] << 24)
                ptn_frame_period_us = readbytes[12] | (readbytes[13] << 8) | (readbytes[14] << 16) | (
                        readbytes[15] << 24)
                self.g_ExpLut.append([lut, exp_time_us, ptn_frame_period_us])
            else:
                return -1
        # Close MailBox
        if self.dlp350_CloseMailBox() < 0:
            return -1
        return 0

    def dlp350_GetImageLut(self, numEntries):
        """
        (USB: CMD2: 0x1A, CMD3: 0x34)
        This API reads the image LUT from the DLPC350 controller.
        :param numEntries:Number of image LUT entries to be read from the controller
        :return: A list contain image LUT
        -1 = Fail
        """
        if numEntries > 64 or numEntries < 1:
            print('Number of entries is out of limit. ')
            return -1
        # Open Mailbox
        imglut_list, lut_left = [], numEntries
        if self.dlp350_OpenMailBox(1) < 0:
            return -1
        if self.dlp350_SetMBoxAddr(0) < 0:
            return -1
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['MBOX_DATA'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            imglut_list = readbytes[4:min(lut_left, 60) + 4]
            lut_left -= 60
        else:
            print('Get ImageLut failed. ')
            self.dlp350_CloseMailBox()
            return -1
        while lut_left > 0:
            readbytes = self.dlp350_continueRead()
            if type(readbytes) is list:
                imglut_list = imglut_list + readbytes[4:min(lut_left, 60) + 4]
            else:
                return -1
        if self.dlp350_CloseMailBox() < 0:
            return -1
        return imglut_list

    def dlp350_GetVarExpImageLut(self, numEntries):
        """
        (USB: CMD2: 0x1A, CMD3: 0x34)
        This API reads the image LUT from the DLPC350 controller.
        :param numEntries: Number of image LUT entries to be read from the controller
        :return:
        """
        VarExpimglut_list, lut_left = [], numEntries
        if numEntries > 256 or numEntries < 1:
            print('Number of entries is out of limit. ')
            return -1
        # Open Mail box
        if self.dlp350_OpenMailBox(1) < 0:
            return -1
        if self.dlp350_SetVarExpMboxAddr(0) < 0:
            return -1
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['MBOX_EXP_DATA'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            VarExpimglut_list = readbytes[4:min(lut_left, 60) + 4]
            lut_left -= 60
        else:
            print('Get ImageLut failed. ')
            self.dlp350_CloseMailBox()
            return -1
        while lut_left > 0:
            readbytes = self.dlp350_continueRead()
            if type(readbytes) is list:
                VarExpimglut_list = VarExpimglut_list + readbytes[4:min(lut_left, 60) + 4]
            else:
                return -1
        if self.dlp350_CloseMailBox() < 0:
            return -1
        return VarExpimglut_list

    def dlp350_SetPatternTriggerMode(self, trigMode=1):
        """
        (USB: CMD2: 0x1A, CMD3: 0x23)
        The Pattern Trigger Mode Selection command selects between one of the three pattern Trigger Modes.
        Before executing this command, stop the current pattern sequence. After executing this command, send
        the Validation command (I2C: 0x7D or USB: 0x1A1A) once before starting the pattern sequence.
        :param trigMode:  0 = Pattern Trigger Mode 0: VSYNC serves to trigger the pattern display sequence. Default is Trigger mode 1
                          1 = Pattern Trigger Mode 1: Internally or Externally (through TRIG_IN1 and TRIG_IN2) generated trigger.
                          2 = Pattern Trigger Mode 2: TRIG_IN_1 alternates between two patterns,while TRIG_IN_2 advances to the next pair of patterns.
                          3 = Pattern Trigger Mode 3: Internally or externally generated trigger for Variable Exposure display sequence.
                          4 = Pattern Trigger Mode 4: VSYNC triggered for Variable Exposure display sequence.
        :return:
        """
        cmd_data = [trigMode]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['PAT_TRIG_MODE'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 set pattern trigger mode failed.')
        return -1

    def dlp350_GetPatternTriggerMode(self):
        """

        :return:
        """
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['PAT_TRIG_MODE'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if Set_Debug_info:
                print("Dlp350 Pattern trigger mode is %d" % readbytes[4])
            return readbytes[4]
        print('Dlp350 get pattern trigger mode failed.')
        return -1

    def dlp350_SetPatternDisplay(self, action=0):
        """
        (USB: CMD2: 0x1A, CMD3: 0x24)
         This API starts or stops the programmed patterns sequence.
        :param action: Pattern Display Start/Stop Pattern Sequence
                        0 = Stop Pattern Display Sequence. The next "Start" command will
                            restart the pattern sequence from the beginning.
                        1 = Pause Pattern Display Sequence. The next "Start" command will
                            start the pattern sequence by re-displaying the current pattern in the sequence.
                        2 = Start Pattern Display Sequence
        :return:
        """
        cmd_data = [action]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['PAT_START_STOP'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 set pattern sequence start/stop mode %d failed.' % action)
        return -1

    def dlp350_GetPatternDisplay(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['PAT_START_STOP'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if Set_Debug_info:
                print("Dlp350 pattern sequence start/stop is at mode %d" % readbytes[4])
            return readbytes[4]
        print('Dlp350 get pattern sequence start/stop mode failed.')
        return -1

    def dlp350_SetPatternConfig(self, numLutEntries=9, repeat=False, numPatsForTrigOut2=9, numImages=1):
        """
        (USB: CMD2: 0x1A, CMD3: 0x31)
        This API controls the execution of patterns stored in the lookup table.
        Before using this API, stop the current pattern sequence using dlp_SetPatternDisplay() API
        After calling this API, send the Validation command using the API dlp350_ValidatePatLutData() before starting the pattern sequence
        :param numLutEntries: Number of LUT entries(1~128)
        :param repeat: 0 = execute the pattern sequence once; 1 = repeat the pattern sequence.
        :param numPatsForTrigOut2: Number of patterns to display(range 1 through 256).
        :param numImages:Number of Image Index LUT Entries(range 1 through 64).
                    This Field is irrelevant for Pattern Display Data Input Source set to a value other than internal.
        :return:
        """
        if numLutEntries < 1 or numLutEntries > 128:
            return -1
        if numPatsForTrigOut2 < 1 or numPatsForTrigOut2 > 256:
            return -1
        if numImages < 1 or numImages > 64:
            return -1
        self.pat_valid = False
        cmd_data = [numLutEntries - 1, int(repeat), numPatsForTrigOut2 - 1, numImages - 1]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['PAT_CONFIG'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Dlp350 set pattern config failed.')
        return -1

    def dlp350_GetPatternConfig(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['PAT_CONFIG'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if Set_Debug_info:
                print('Pattern Config Setting:')
                print('NumLutEntries: %d, Repeat Pattern: %d, NumPatToDis: %d, NumImages: %d'
                      % (readbytes[4] + 1, readbytes[5], readbytes[6] + 1, readbytes[7] + 1))
            return readbytes[4] + 1, readbytes[5], readbytes[6] + 1, readbytes[7] + 1
        print('Dlp350 Get pattern Config setting failed')
        return -1

    def dlp350_SetVarExpPatternConfig(self, numLutEntries=9, numPatsForTrigOut2=9, numImages=1, repeat=False):
        """
        (USB: CMD2: 0x1A, CMD3: 0x40)
        This API controls the execution of patterns stored in the lookup table.
        Before using this API, stop the current pattern sequence using dlp_SetPatternDisplay() API
        After calling this API, send the Validation command using the API dlp350_ValidatePatLutData() before starting the pattern sequence
        :param numLutEntries: Number of LUT entries(1~1824)
        :param numPatsForTrigOut2:Number of patterns to display(range 1 through 1824).
        :param numImages:Number of Image Index LUT Entries(range 1 through 256).
        :param repeat:0 = execute the pattern sequence once; 1 = repeat the pattern sequence.
        :return:
        """
        if numLutEntries < 1 or numLutEntries > 1824:
            return -1
        if numPatsForTrigOut2 < 1 or numPatsForTrigOut2 > 1824:
            return -1
        if numImages < 1 or numImages > 256:
            return -1
        self.pat_valid = False
        cmd_data = [(numLutEntries - 1) & 0xFF, ((numLutEntries - 1) >> 8), (numPatsForTrigOut2 - 1) & 0xFF,
                    (numPatsForTrigOut2 - 1) >> 8, (numImages - 1) & 0xFF, int(repeat)]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['EXP_PAT_CONFIG'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print("Dlp350 set variable exposure LUT control failed")
        return -1

    def dlp350_GetVarExpPatternConfig(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['EXP_PAT_CONFIG'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            numLutEntries = readbytes[4] + (readbytes[5] << 8) + 1
            numPatsForTrigOut2 = readbytes[6] + (readbytes[7] << 8) + 1
            numImages = readbytes[8] + 1
            repeat = readbytes[9]
            if Set_Debug_info:
                print('Variable Exposure Pattern Config Setting:')
                print('NumLutEntries: %d, , NumPatToDis: %d, NumImages: %d Repeat Pattern: %d.'
                    % (numLutEntries, numPatsForTrigOut2, numImages, repeat))
            return numLutEntries, numPatsForTrigOut2, numImages, repeat
        print("Dlp350 Get variable exposure pattern config setting is failed.")
        return -1

    def dlp350_SetExposure_FramePeriod(self, exposurePeriod=0x4010, framePeriod=0x411A):
        """
        USB: CMD2: 0x1A, CMD3: 0x29
        The Pattern Display Exposure and Frame Period dictates the time a pattern is exposed and the frame
        period. Either the exposure time must be equivalent to the frame period, or the exposure time must be
        less than the frame period by 230 microseconds. Before executing this command, stop the current pattern
        sequence. After executing this command, call dlp350_ValidatePatLutData() API before starting the pattern sequence.
        :param exposurePeriod:Exposure time in microseconds.
        :param framePeriod:Frame period in microseconds.
        :return:
        """
        if exposurePeriod < 0 or framePeriod < 0:
            print("Exposure and Frame Period must >= 0us")
            return -1
        if not (exposurePeriod == framePeriod or exposurePeriod < (framePeriod - 230)):
            print("Wrong exposure and frame period setting. ")
            return -1
        cmd_data = [exposurePeriod & 0xFF, (exposurePeriod >> 8) & 0xFF, (exposurePeriod >> 16) & 0xFF,
                    ((exposurePeriod >> 24) & 0xFF),
                    framePeriod & 0xFF, (framePeriod >> 8) & 0xFF, (framePeriod >> 16) & 0xFF,
                    (framePeriod >> 24) & 0xFF]
        self.pat_valid = False
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['PAT_EXPO_PRD'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print("Dlp350 Set exposure frame period error.")
        return -1

    def dlp350_GetExposure_FramePeriod(self):
        """

        :return: exposurePeriod, framePeriod
        """
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['PAT_EXPO_PRD'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            exposurePeriod = readbytes[4] | (readbytes[5] << 8) | (readbytes[6] << 16) | (readbytes[7] << 24)
            framePeriod = readbytes[8] | (readbytes[9] << 8) | (readbytes[10] << 16) | (readbytes[11] << 24)
            if Set_Debug_info:
                print("Pattern Exposure Time is %dus, Frame Period is %dus." % (exposurePeriod, framePeriod))
            return exposurePeriod, framePeriod
        print('DLP350 get pattern exposure time and frame period is failed')
        return -1

    def dlp350_SetTrig1OutConfig(self, invert=0, rising=0xBB, failing=0xBB):
        """
        (USB: CMD2: 0x1A, CMD3: 0x1D)
        This API sets the polarity, rising edge delay, and falling edge delay of the DLPC350's TRIG_OUT_1 signal.
        The delays are compared to when the pattern is displayed on the DMD. Before executing this command,
        stop the current pattern sequence. After executing this command, call dlp350_ValidatePatLutData() API before starting the pattern sequence.
        :param invert: 0 = active High signal; 1 = Active Low signal
        :param rising: rising edge delay control. Each bit adds 107.2 ns
            0x00 = -20.05 μs, 0x01 = -19.9428 μs, ......0xBB=0.00 μs, ......, 0xD4 = +2.68 μs, 0xD5 = +2.787 μs
        :param failing: falling edge delay control. Each bit adds 107.2 ns (This field is not applcable for TRIG_OUT_2)
            0x00 = -20.05 μs, 0x01 = -19.9428 μs, ......0xBB=0.00 μs, ......, 0xD4 = +2.68 μs, 0xD5 = +2.787 μs
        :return:
        """
        self.pat_valid = False
        cmd_data = [invert, rising, failing]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['TRIG_OUT1_CTL'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print("Dlp350 Set Trigger Out 1 control failed.")
        return -1

    def dlp350_SetTrig2OutConfig(self, invert=0, rising=0xBB):
        """
        (USB: CMD2: 0x1A, CMD3: 0x1E)
        This API sets the polarity, rising edge delay, and falling edge delay of the DLPC350's TRIG_OUT_2 signal.
        The delays are compared to when the pattern is displayed on the DMD. Before executing this command,
        stop the current pattern sequence. After executing this command, call dlp350_ValidatePatLutData() API before starting the pattern sequence.
        :param invert: 0 = active High signal; 1 = Active Low signal
        :param rising: rising edge delay control. Each bit adds 107.2 ns From -20.05 to 7.29us
        :return:
        """
        self.pat_valid = False
        cmd_data = [invert, rising]
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['TRIG_OUT2_CTL'], data=cmd_data, ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print("Dlp350 Set Trigger Out 2 control failed.")
        return -1

    def dlp350_GetTrig1OutConfig(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['TRIG_OUT1_CTL'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            invert = readbytes[4]
            rising = readbytes[5]
            failing = readbytes[6]
            if Set_Debug_info:
                print("Trigger Out1 Config: ")
                print("Invert: %d, Rising_Delay_Byte: %#x, Falling_Delay_Byte: %#x" % (invert, rising, failing))
            return invert, rising, failing
        print('Dlp350 get Trigger out1 config failed.')
        return -1

    def dlp350_GetTrig2OutConfig(self):
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['TRIG_OUT2_CTL'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            invert = readbytes[4]
            rising = readbytes[5]
            if Set_Debug_info:
                print("Trigger Out2 Config: ")
                print("Invert: %d, Rising_Delay_Byte: %#x," % (invert, rising))
            return invert, rising
        print('Dlp350 get Trigger out2 config failed.')
        return -1

    def dlp350_StartPatLutValidate(self):
        """
        (USB: CMD2: 0x1A, CMD3: 0x1A)
        This API checks the programmed pattern display modes and indicates any invalid settings.
        :return:
        """
        msg_pkg = self.dlp350_PrepWriteCMD(CMD=Cmdlist['LUT_VALID'], data=[0], ackRequired=False)
        if self.dlp350_SendMessage(msg_pkg, ackRequired=False) > 0:
            return 0
        print('Start Validation status failed.')
        return -1

    def dlp350_CheckPatLutValidate(self):
        """
        Check self.pat_valid
        :return:0 = validation complete.
                -1 = validation failed or busy
        """
        msg_pkg = self.dlp350_PrepReadCMD(CMD=Cmdlist['LUT_VALID'])
        readbytes = self.dlp350_SendMessage(msg_pkg)
        if type(readbytes) is list:
            if (readbytes[4] & 0x80) == 0:
                if Set_Debug_info:
                    print("Validation complete. Valid byte is %#x" % readbytes[4])
                if readbytes[4] == 0:
                    self.pat_valid = True
                    if Set_Debug_info:
                        print('Pattern setting are all valid.')
                else:
                    print('Pattern Setting has issue. Please check below info:')
                    self.pat_valid = False
                    if readbytes[4] & BIT0 == 1:
                        print('     Select exposure or frame period settings are invalid. ')
                    elif (readbytes[4] & BIT1) >> 1 == 1:
                        print('     Select pattern numbers in LUT are invalid. ')
                    elif (readbytes[4] & BIT2) >> 2 == 1:
                        print('     Warning, continuous Trigger Out1 request or overlapping black sectors ')
                    elif (readbytes[4] & BIT3) >> 3 == 1:
                        print('     Warning, post vector is not inserted prior to external triggered vector. ')
                    elif (readbytes[4] & BIT4) >> 4 == 1:
                        print('     Warning, frame period or exposure difference is less than 230 µsec. ')
                return 0
            else:
                if Set_Debug_info:
                    print('DLP350 is busy validating')
                self.pat_valid = False
                return -1
        print('Get validation status failed.')
        return -1


class HidMessageStruct:
    def __init__(self, CMD=Cmdlist['STATUS_HW'], ackRequired=True):
        self.dest = 0x0  # 0-ProjCtrl; 1-RFC; 0x7-Debugger
        self.nack = 0
        self.reply = int(ackRequired)  # 0-host don't require a reply. 1-Host needs a reply
        self.rw = 1  # Write = 0, Read = 1
        self.seq = 0  # if data longer than 64bytes, then seq represent package numbers
        self.cmd = CMD  # Default check HW status
        self.cmddata = [0 for i in range(58)]
        self.data_len = 2 + CMD[2]

    def Message_to_Data(self):
        head = [0x0, self.rw << 7 ^ self.reply << 6 ^ self.nack << 5 ^ self.dest, self.seq, self.data_len % 256,
                self.data_len >> 8]
        pkg_seq = []
        maxDataSize = USB_MAX_PACKET_SIZE + 1 - len(head)
        dataByteSent = min(self.data_len, maxDataSize)
        if dataByteSent >= self.data_len:
            pkg_seq.append(head + [self.cmd[1], self.cmd[0]] + self.cmddata)
            return pkg_seq
        else:
            assert len(self.cmddata) >= (self.data_len - 2), 'CMD Data contain not enought bytes as cmd required. '
            pkg_seq.append(head + [self.cmd[1], self.cmd[0]] + self.cmddata[0:dataByteSent - 2])
            while (dataByteSent < self.data_len):
                if (dataByteSent + USB_MAX_PACKET_SIZE) > self.data_len:
                    pkg_seq.append([0] + self.cmddata[dataByteSent - 2:self.data_len - 2] + [0 for i in range(
                        USB_MAX_PACKET_SIZE - (self.data_len - dataByteSent))])
                    return pkg_seq
                else:
                    pkg_seq.append([0] + self.cmddata[dataByteSent - 2:dataByteSent - 2 + USB_MAX_PACKET_SIZE])
                dataByteSent += USB_MAX_PACKET_SIZE
            return pkg_seq

class Pattern_Entry:
    """
    Pattern entry class, contains each pattern config.
    """

    def __init__(self, id=0, pat_num=0, bit_depth=1, invert=False, insert_black=True,
                 image_index=0, pat_exp_time=300000, frame_period=301000):
        self.id = id
        self.triggerType = 0  # internal trigger
        self.pat_num = pat_num
        self.bit_depth = bit_depth
        self.invert = invert
        self.insert_black = insert_black
        self.image_index = image_index
        self.buffer_swap = False
        self.pat_exp_time = pat_exp_time
        self.frame_period = frame_period

class PyLCR4500:
    def __init__(self):
        self.lcr4500 = Dlp350_api()
        self.fw_Tag = None
        self.num_of_image = 1
        self.image_lut = []
        self.pat_sequence = []
        self.varPat_sequence = []
        self.repeat_pattern = False
        self.pattern_sequence_prepared = False
        self.previous_sequence_start_ = 0
        self.previous_sequence_patterns_ = 0
        self.previous_sequence_repeat_ = False
        self.pat_exp_time = 300000
        self.frame_period = 301000
        self.standby = False  # If projector in standby mode


        # Connect Projector.
        self.connect()
        # Setup Projector.
        assert self.Set_Projector(led_current=Config.led_current, pat_exp_time=Config.pat_exp_time,
                                    frame_period=Config.frame_period) >= 0, 'Projector setup failed.'
        # Generate pattern sequence. For Grey-Phase patterns, there will be 10 1-bit patterns followed by 4 8-bit patterns.
        self.pat_sequence = GeneratePatternSequence(pat_exp_time=Config.pat_exp_time,
                                                        frame_period=Config.frame_period)
        # Prepare projector pattern.
        assert self.Prepare_Projector_Pattern() >= 0, 'Prepare Projector pattern failed.'
        print('Prepare Pattern success!, Total pattern number is: %d' % len(self.pat_sequence))

    def connect(self):
        if self.lcr4500.usb.isUSBConnected:
            self.lcr4500.dlp350_disconnect()
        self.lcr4500.dlp350_connect()
        if self.lcr4500.dlp350_SetPowerMode(False) < 0:
            return -1
        self.standby = False
        ret = self.lcr4500.dlp350_ReadFwTag()
        if ret == -1:
            print('Get Projector firmware tag issue. ')
            return -1
        else:
            self.fw_Tag = ret
            print('Projector connected successfully.', end=' ')
            print("FW Tag Version is: %s" % ret)
            return 0

    def disconnect(self):
        self.lcr4500.dlp350_disconnect()

    def Set_Projector(self, led_current=100, pat_exp_time=300000, frame_period=301000):
        if not self.lcr4500.usb.isUSBConnected:
            print('Projector is not connected. ')
            return -1
        if self.standby:
            if self.lcr4500.dlp350_SetPowerMode(False) < 0:
                return -1
            self.standby = False
        # Stop pattern display.
        if self.lcr4500.dlp350_SetPatternDisplay(0) < 0:
            return -1
        if self.lcr4500.dlp350_SetMode(SLmode=True) < 0:  # Set to pattern display mode.
            return -1
        time.sleep(insert_delay)
        if self.lcr4500.dlp350_SetInputSource(source=2,
                                              port_width=1) < 0:  # Set projector input source as internal flash.
            return -1
        if self.lcr4500.dlp350_SetLedCurrent(led_current,
                                             led_current,
                                             led_current) < 0:  # Set default LED current.
            return -1
        if self.lcr4500.dlp350_SetPatternDisplayMode(False) < 0:  # Set pattern fetch from internal image.
            return -1
        # Set trigger mode for internally triggered for variable exposure
        if self.lcr4500.dlp350_SetPatternTriggerMode(3) < 0:  # For pattern trigger mode, set to 3
            return -1
        if self.lcr4500.dlp350_SetExposure_FramePeriod(exposurePeriod=pat_exp_time,
                                                       framePeriod=frame_period) < 0:
            return -1
        if self.lcr4500.dlp350_SetLedEnables(all_led=True, red=False, grn=False, blu=False) < 0:  # Enable all les.
            return -1
        if self.lcr4500.dlp350_GetMode() != 1:
            print('Projector Set mode to pattern display mode failed.')
            return -1
        if self.lcr4500.dlp350_GetInputSource() != (2, 1):
            print('Projector Set input source to flash failed. ')
            return -1
        ret = self.lcr4500.dlp350_GetLedCurrents()
        if ret == -1:
            print('Get LED Current failed.')
            return -1
        elif ret[2] != Config.led_current:
            print('Projector LED Set current failed. ')
            return -1
        if self.lcr4500.dlp350_GetPatternDisplayMode() != 3:
            print('Projector Set Pattern source fetch from flash failed.')
            return -1
        if self.lcr4500.dlp350_GetPatternTriggerMode() != 3:
            print('Projector Set trigger mode as internal failed.')
        ret = self.lcr4500.dlp350_GetExposure_FramePeriod()
        if ret == -1:
            print('Get Projector Pattern exposure/period time setting failed.')
            return -1
        elif ret != (pat_exp_time, frame_period):
            print('Set Desire pattern exposure/period time failed.')
            return -1
        self.pat_exp_time, self.frame_period = ret
        self.num_of_image = self.lcr4500.dlp350_GetNumImagesInFlash()
        print('Projector setting success. ')
        return 0

    def ProjectSolidWhitePattern(self):
        if not self.lcr4500.usb.isUSBConnected:
            print('Projector is not connected. ')
            return -1
        if self.standby:
            print('Projector in standby mode, needs to wakeup first.')
            return -1
        white_pattern = Pattern_Entry(insert_black=False, pat_exp_time=300000, frame_period=300000)
        white_pattern.pat_num = Pat_Num_MONO_1BPP['BLACK']
        white_pattern.invert = True
        if self.CreatSendSequence([white_pattern], repeat=True) < 0:
            return -1
        self.previous_sequence_start_ = 0
        self.previous_sequence_patterns_ = 0
        self.previous_sequence_repeat_ = True
        # Start projecting
        if self.lcr4500.dlp350_SetPatternDisplay(action=2) < 0:
            return -1
        return 0

    def ProjectSolidBlackPattern(self):
        if not self.lcr4500.usb.isUSBConnected:
            print('Projector is not connected. ')
            return -1
        if self.standby:
            print('Projector in standby mode, needs to wakeup first.')
            return -1
        black_pattern = Pattern_Entry(insert_black=False, pat_exp_time=300000, frame_period=300000)
        black_pattern.pat_num = Pat_Num_MONO_1BPP['BLACK']
        black_pattern.invert = False
        if self.CreatSendSequence([black_pattern], repeat=True) < 0:
            return -1
        self.previous_sequence_start_ = 0
        self.previous_sequence_patterns_ = 0
        self.previous_sequence_repeat_ = True
        # Start projecting
        if self.lcr4500.dlp350_SetPatternDisplay(action=2) < 0:
            return -1
        return 0

    def CreatSendSequence(self, sequence, repeat=False):
        """
        Private method.
        :param sequence:
        :param repeat:
        :return:
        """
        if len(sequence) == 0 or len(sequence) > 255:
            print('Sequence is empty or exceed 255 ')
        # Clear up image lut.
        self.image_lut = []
        previous_image_index = None
        # validate each pattern timing setting:
        for index in range(len(sequence)):
            if self.Pattern_Setting_Valid(sequence[index]) < 0:
                return -1
            if (sequence[index].image_index == previous_image_index) and (index != 0):
                sequence[index].buffer_swap = False
            else:
                sequence[index].buffer_swap = True
                previous_image_index = sequence[index].image_index
                self.image_lut.append(sequence[index].image_index)
        # To do: Add image load time checking if images more than 2.
        if not self.lcr4500.usb.isUSBConnected:
            print('Projector is not connected. ')
            return -1
        if self.standby:
            print('Projector in standby mode, needs to wakeup first.')
            return -1
        # Stop the display.
        if self.lcr4500.dlp350_SetPatternDisplay(0) < 0:
            return -1
        # Clear dlp pattern buffer.
        self.lcr4500.dlp350_ClearExpLut()
        for pat in sequence:
            if self.lcr4500.dlp350_AddToExpLut(TrigType=0,
                                               PatNum=pat.pat_num,
                                               BitDepth=pat.bit_depth,
                                               LEDSelect=7,
                                               InvertPat=pat.invert,
                                               InsertBlack=pat.insert_black,
                                               BufSwap=pat.buffer_swap,
                                               trigOutPrev=False,
                                               exp_time_us=pat.pat_exp_time,
                                               ptn_frame_period_us=pat.frame_period) < 0:
                return -1
        # Send image lut
        if self.lcr4500.dlp350_SendVarExpImageLut(self.image_lut, len(self.image_lut)) == -1:
            print('Send ImageLut failed')
            return -1
        time.sleep(0.01)
        if self.lcr4500.dlp350_SendVarExpPatLut() == -1:
            print('Send Pattern lut failed')
            return -1
        if Set_Debug_info:
            print('Lut upload complete. check validation')
        # Set Pattern Config
        time.sleep(0.01)
        if self.lcr4500.dlp350_SetVarExpPatternConfig(numLutEntries=len(sequence), numPatsForTrigOut2=len(sequence),
                                                      numImages=len(self.image_lut), repeat=repeat) < 0:
            return -1
        time.sleep(0.01)
        if self.lcr4500.dlp350_StartPatLutValidate() == -1:
            return -1
        time.sleep(0.01)
        valid_check_intend = 10
        while valid_check_intend >= 0:
            if self.lcr4500.dlp350_CheckPatLutValidate() >= 0:
                break
            time.sleep(0.1)
            valid_check_intend -= 1
        if self.lcr4500.pat_valid:
            return 0
        return -1

    def Pattern_Setting_Valid(self, pat):
        """
        Private method. Validate Pattern setting.
        :param pat: Pattern_Entry object.
        :return: 0 = PASS
                -1 = FAIL
        """
        if pat.bit_depth == 1:
            if pat.pat_num < 0 or pat.pat_num > 24:
                print('Pat Number is invalid.')
                return -1
            if pat.pat_exp_time < MINIMUM_EXPOSURE_TIME['MONO_1BPP'] or pat.pat_exp_time > 2000000:
                print('Pattern exposure time is not valid, has to be larger than %dus' % MINIMUM_EXPOSURE_TIME[
                    'MONO_1BPP'])
                return -1
        elif pat.bit_depth == 2:
            if pat.pat_num < 0 or pat.pat_num > 11:
                print('Pat Number is invalid.')
                return -1
            if pat.pat_exp_time < MINIMUM_EXPOSURE_TIME['MONO_2BPP'] or pat.pat_exp_time > 2000000:
                print('Pattern exposure time is not valid, has to be larger than %dus' % MINIMUM_EXPOSURE_TIME[
                    'MONO_2BPP'])
                return -1
        elif pat.bit_depth == 3:
            if pat.pat_num < 0 or pat.pat_num > 7:
                print('Pat Number is invalid.')
                return -1
            if pat.pat_exp_time < MINIMUM_EXPOSURE_TIME['MONO_3BPP'] or pat.pat_exp_time > 2000000:
                print('Pattern exposure time is not valid, has to be larger than %dus' % MINIMUM_EXPOSURE_TIME[
                    'MONO_3BPP'])
                return -1
        elif pat.bit_depth == 4:
            if pat.pat_num < 0 or pat.pat_num > 5:
                print('Pat Number is invalid.')
                return -1
            if pat.pat_exp_time < MINIMUM_EXPOSURE_TIME['MONO_4BPP'] or pat.pat_exp_time > 2000000:
                print('Pattern exposure time is not valid, has to be larger than %dus' % MINIMUM_EXPOSURE_TIME[
                    'MONO_4BPP'])
                return -1
        elif pat.bit_depth == 5:
            if pat.pat_num < 0 or pat.pat_num > 3:
                print('Pat Number is invalid.')
                return -1
            if pat.pat_exp_time < MINIMUM_EXPOSURE_TIME['MONO_5BPP'] or pat.pat_exp_time > 2000000:
                print('Pattern exposure time is not valid, has to be larger than %dus' % MINIMUM_EXPOSURE_TIME[
                    'MONO_5BPP'])
                return -1
        elif pat.bit_depth == 6:
            if pat.pat_num < 0 or pat.pat_num > 3:
                print('Pat Number is invalid.')
                return -1
            if pat.pat_exp_time < MINIMUM_EXPOSURE_TIME['MONO_6BPP'] or pat.pat_exp_time > 2000000:
                print('Pattern exposure time is not valid, has to be larger than %dus' % MINIMUM_EXPOSURE_TIME[
                    'MONO_6BPP'])
                return -1
        elif pat.bit_depth == 7:
            if pat.pat_num < 0 or pat.pat_num > 2:
                print('Pat Number is invalid.')
                return -1
            if pat.pat_exp_time < MINIMUM_EXPOSURE_TIME['MONO_7BPP'] or pat.pat_exp_time > 2000000:
                print('Pattern exposure time is not valid, has to be larger than %dus' % MINIMUM_EXPOSURE_TIME[
                    'MONO_7BPP'])
                return -1
        elif pat.bit_depth == 8:
            if pat.pat_num < 0 or pat.pat_num > 2:
                print('Pat Number is invalid.')
                return -1
            if pat.pat_exp_time < MINIMUM_EXPOSURE_TIME['MONO_8BPP'] or pat.pat_exp_time > 2000000:
                print('Pattern exposure time is not valid, has to be larger than %dus' % MINIMUM_EXPOSURE_TIME[
                    'MONO_8BPP'])
                return -1
        else:
            print('Pat bit depth is invalid.')
            return -1
        # If pattern exposure time is not equal to frame period, there must be fill black between two pattern.
        if pat.pat_exp_time != pat.frame_period:
            pat.insert_black = True
        if pat.image_index > (self.num_of_image - 1) or pat.image_index < 0:
            print('Pattern index exceed image that store in projector.')
            return -1
        return 0

    def Prepare_Projector_Pattern(self):
        """
        Prepare Projector Pattern.
        :return:  0 = PASS
                 -1 = FAIL
        """
        if len(self.pat_sequence) == 0:
            print('Pattern sequence list is empty.')
            return -1
        self.previous_sequence_start_ = 0
        self.previous_sequence_patterns_ = len(self.pat_sequence)
        self.previous_sequence_repeat_ = False
        if self.CreatSendSequence(self.pat_sequence, self.repeat_pattern) < 0:
            return -1
        self.pattern_sequence_prepared = True
        return 0

    # def Download_PatLut(self):
    #     """
    #     Download patttern lut from project, will be stored in self.lcr4500.g_Patlut.
    #     :return: 0 = PASS
    #             -1 = FAIL
    #     """
    #     if not self.lcr4500.usb.isUSBConnected:
    #         print('Projector is not connected. ')
    #         return -1
    #     if self.lcr4500.dlp350_SetPatternDisplay(0) < 0:
    #         return -1
    #     self.lcr4500.dlp350_GetPatLut(len(self.pat_sequence))
    #     time.sleep(10)
    #     ret = self.lcr4500.dlp350_GetImageLut(len(self.pat_sequence))
    #     if ret == -1:
    #         print('Get Image Lut error.')
    #         return -1
    #     print(ret)

    def ProjectPatternOnce(self):
        """
        Display pattern sequence that store in project lut.
        :return:  0=PASS
                 -1=Fail
        """
        return self.StartPatternSequence(start=0, patterns=len(self.pat_sequence), repeat=False)
    
    def scan_one_pattern(self, index):
        return self.StartPatternSequence(start=index, patterns=1, repeat=False)

    def StartPatternSequence(self, start=0, patterns=1, repeat=False):
        """
        Displays a previously prepared pattern sequence
        :param start: Patten index
        :param patterns: Numbers of pattern to be display.
        :param repeat: Whether repeat patterns.
        :return: 0 = PASS
                -1 = FAIL
        """
        if not self.lcr4500.usb.isUSBConnected:
            print('Projector is not connected. ')
            return -1
        if not self.pattern_sequence_prepared:
            print('Projector Pattern Sequence is not prepared.')
            return -1
        if self.standby:
            print('Projector in standby mode, needs to wakeup first.')
            return -1
        # Stop pattern display first
        if self.lcr4500.dlp350_SetPatternDisplay(0) < 0:
            return -1
        if (start + patterns) > len(self.pat_sequence) or start < 0 or patterns <= 0:
            print('Pattern sequence index is out of range.')
            return -1
        if (start != self.previous_sequence_start_) or \
                (patterns != self.previous_sequence_patterns_) or \
                (repeat != self.previous_sequence_repeat_):
            # Create the sequence.
            sequence_tmp = self.pat_sequence[start:start + patterns]
            if self.CreatSendSequence(sequence_tmp, repeat=repeat) < 0:
                return -1
            self.previous_sequence_start_ = start
            self.previous_sequence_patterns_ = patterns
            self.previous_sequence_repeat_ = repeat
            # Display pattern start.
            print('Start Pattern sequences.')
            time.sleep(0.01)
            if self.lcr4500.dlp350_SetPatternDisplay(2) < 0:
                return -1
        else:
            print('Start Pattern sequences.')
            if self.lcr4500.dlp350_SetPatternDisplay(2) < 0:
                return -1
        return 0

    def StopPatternSequence(self):
        """
        Stop current pattern display.
        :return: 0 = PASS
                -1 = FAIL
        """
        if not self.lcr4500.usb.isUSBConnected:
            print('Projector is not connected. ')
            return -1
        if self.standby:
            print('Projector in standby mode, needs to wakeup first.')
            return -1
        if self.lcr4500.dlp350_SetPatternDisplay(0) < 0:
            print('Stop Pattern display failed.')
            return -1
        return 0

    def SetLEDCurrent(self, led_current):
        if self.lcr4500.dlp350_SetPatternDisplay(0) < 0:
                return -1
        if self.lcr4500.dlp350_SetLedCurrent(led_current,
                                            led_current,
                                            led_current) < 0:  # Set default LED current.
            print("Set LED current fail")
            return -1

    def UpdateProjectorSetting(self, pat_exp_time=10000, frame_period=11000, led_current=100):
        """
        Update projector setting from Config
        :return: 0 = PASS
                -1 = FAIL
        """
        if not self.lcr4500.usb.isUSBConnected:
            print('Projector is not connected. ')
            return -1
        if self.standby:
            print('Projector in standby mode, needs to wakeup first.')
            return -1
        # Stop Current projection
        if self.lcr4500.dlp350_SetPatternDisplay(0) < 0:
            return -1
        if self.lcr4500.dlp350_SetLedCurrent(led_current,
                                             led_current,
                                             led_current) < 0:  # Set default LED current.
            return -1
        if self.lcr4500.dlp350_SetExposure_FramePeriod(exposurePeriod=pat_exp_time,
                                                       framePeriod=frame_period) < 0:
            return -1
        # Once change exposure, all pattern has to be revalidated.
        self.pattern_sequence_prepared = False
        ret = self.lcr4500.dlp350_GetLedCurrents()
        if ret == -1:
            print('Get LED Current failed.')
            return -1
        elif ret[2] != Config.led_current:
            print('Projector LED Set current Failed')
            return -1
        ret = self.lcr4500.dlp350_GetExposure_FramePeriod()
        if ret == -1:
            print('Get Projector Pattern exposure/period time setting failed.')
            return -1
        elif ret != (pat_exp_time, frame_period):
            print('Set Desire pattern exposure/period time failed.')
            return -1
        self.pat_exp_time, self.frame_period = ret
        if len(self.pat_sequence) == 0:
            print('Pattern sequence list is empty.')
            return -1
        for pat in self.pat_sequence:
            pat.pat_exp_time = pat_exp_time
            pat.frame_period = frame_period
        return 0

    def Set_Projector_to_Standby(self):
        if not self.lcr4500.usb.isUSBConnected:
            print('Projector is not connected. ')
            return -1
        # Stop Current projection
        if self.lcr4500.dlp350_SetPatternDisplay(0) < 0:
            return -1
        # Disable all leds.
        if self.lcr4500.dlp350_SetLedEnables(False, False, False, False) < 0:
            return -1
        # Set Projector to standby mode.
        # if self.lcr4500.dlp350_SetPowerMode(standby=True) < 0:
        #     return -1
        self.standby = True
        return 0

    def Wakeup_Projector(self):
        # Wakeup projector.
        if self.lcr4500.dlp350_SetPowerMode(False) < 0:
            return -1
        time.sleep(0.01)
        # Make LEDs are control by sequencer.
        if self.lcr4500.dlp350_SetLedEnables(all_led=True) < 0:
            return -1
        self.standby = False
        return 0

def GeneratePatternSequence(pat_exp_time=300000, frame_period=301000):
    """
    Generate grey-phase pattern sequence.
    :return: A list of grey-phase pattern.
    """
    grey_phase_sequence = []
    for i in range(10):
        pat = Pattern_Entry(id=i, pat_num=i, bit_depth=1, invert=False, insert_black=True, image_index=0,
                            pat_exp_time=pat_exp_time, frame_period=frame_period)
        grey_phase_sequence.append(pat)
    pat_phase_1 = Pattern_Entry(id=10, pat_num=2, bit_depth=8, invert=False, insert_black=True, image_index=0,
                                pat_exp_time=pat_exp_time, frame_period=frame_period)
    grey_phase_sequence.append(pat_phase_1)
    for i in range(3):
        pat = Pattern_Entry(id=10 + i, pat_num=i, bit_depth=8, invert=False, insert_black=True, image_index=1,
                            pat_exp_time=pat_exp_time, frame_period=frame_period)
        grey_phase_sequence.append(pat)
    return grey_phase_sequence


if __name__ == '__main__':
    Projector = PyLCR4500()
    while True:
        print('Press 99 to exit.')
        print('Press 1 to Update Projector Setting.')
        print('Press 2 to project white pattern.')
        print('Press 3 to project black pattern.')
        print('Press 4 to project all pattern in pattern sequence.')
        print('Press 5 to project specific pattern.')
        print('Press 6 to project desire patterns.')
        print('Press 7 to stop current pattern.')
        print('Press 8 to set projector standby.')
        print('Press 9 to wakeup projector.')
        print('Press 0 to reconnect project.')
        menu_input = int(input('Select Menu item: '))
        if menu_input == 99:
            Projector.disconnect()
            break
        elif menu_input == 1:
            pat_exp_time = int(input('Input Pattern exposure time(us): '))
            print('Note: If pattern insert black is true, pattern interval time has to be larger than exposure time 230us at least. ')
            frame_period = int(input('Input Pattern interval time(us): '))
            led_current = int(input('Input Led current: '))
            if Projector.UpdateProjectorSetting(pat_exp_time, frame_period, led_current) < 0:
                print('Update Project Setting fail.')
            assert Projector.Prepare_Projector_Pattern() >= 0, 'Prepare Projector pattern failed.'
            continue
        elif menu_input == 2:
            if Projector.ProjectSolidWhitePattern() < 0:
                print('Project Solid white pattern failed.')
            continue
        elif menu_input == 3:
            if Projector.ProjectSolidBlackPattern() < 0:
                print('Project Solid black pattern failed.')
            continue
        elif menu_input == 4:
            if Projector.ProjectPatternOnce() < 0:
                print('Display pattern sequence failed.')
            continue
        elif menu_input == 5:
            print('Pattern index has to be between 0 ~ %d' % (len(Projector.pat_sequence) - 1))
            print('Enter 99 to upper menu.')
            while True:
                pattern_select = int(input('Select Pattern index: '))
                if pattern_select == 99:
                    break
                elif pattern_select < 0 or pattern_select > (len(Projector.pat_sequence) - 1):
                    print('Wrong pattern index input.')
                    continue
                if Projector.StartPatternSequence(pattern_select, 1, True) < 0:
                    print('Display pattern %d failed.' % pattern_select)
            continue
        elif menu_input == 6:
            start = int(input('Select Start pattern: '))
            patterns = int(input('Select Patterns to display: '))
            Projector.StartPatternSequence(start, patterns, False)
            continue
        elif menu_input == 7:
            Projector.StopPatternSequence()
            continue
        elif menu_input == 8:
            Projector.Set_Projector_to_Standby()
            continue
        elif menu_input == 9:
            Projector.Wakeup_Projector()
            continue
        elif menu_input == 0:
            Projector.disconnect()
            Projector.connect()
            continue
        else:
            print('Wrong Menu input. ')
    print("Close projector.")
    Projector.disconnect()
