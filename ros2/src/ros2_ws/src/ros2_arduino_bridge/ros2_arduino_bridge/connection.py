"""
Двустороннее общение по Serial Master - Slave
"""
import time
from enum import Enum
from struct import Struct
from typing import Final
from typing import Optional
from dataclasses import dataclass

import serial.tools.list_ports
from serial import Serial


def find_arduino_port():
    """Автоматически найти порт Arduino"""
    for port in serial.tools.list_ports.comports():
        if 'usb' in port.description.lower():
            return port.device

    return None


class Primitives(Enum):
    """Примитивные типы"""

    i8 = Struct("b")
    """int8_t"""
    u8 = Struct("B")
    """uint8_t"""
    i16 = Struct("h")
    """int16_t"""
    u16 = Struct("H")
    """uint16_t"""
    i32 = Struct("<i")
    """int32_t"""
    u32 = Struct("I")
    """uint32_t"""
    i64 = Struct("q")
    """int64_t"""
    I64 = Struct("Q")
    """unt64_t"""
    f32 = Struct("<f")
    """float"""
    f64 = Struct("d")  # ! Не поддерживается на Arduino
    """double"""

    def pack(self, value: bool | int | float) -> bytes:
        return self.value.pack(value)

    def unpack(self, buffer: bytes) -> bool | int | float:
        return self.value.unpack(buffer)[0]

    def size(self) -> int:
        return self.value.size


@dataclass(frozen=True)
class Data:
    left_delta: int
    right_delta: int
    left_speed: float
    right_speed: float
    data_packer = Struct("hhff")

    @classmethod
    def make(cls, buffer: bytes):
        return cls(*cls.data_packer.unpack(buffer))


class Command:
    """
    Команда по порту

    Имеет свой код (Должен совпадать на slave устройстве)
    Сигнатуру аргументов (Должна совпадать на устройстве)
    """

    def __init__(self, code: int, signature: tuple[Primitives, ...]) -> None:
        self.header: Final[bytes] = Primitives.u8.pack(code)
        self.signature = signature

    def pack(self, *args) -> bytes:
        """
        Скомпилировать команду в набор байт
        :param args: аргументы команды. (Их столько же, и такого же типа, что и сигнатура команды)
        :return:
        """
        return self.header + b"".join(primitive.pack(arg) for primitive, arg in zip(self.signature, args))


class ArduinoConnection:
    """Пример подключения к Arduino с dataминимальным набором команд"""

    def __init__(self, __serial: Serial) -> None:
        self.serial = __serial
        # Команды этого устройства

        self._heartbeat = Command(0x00, ())

        self._set_velocity = Command(0x10, (Primitives.f32, Primitives.f32))
        self._turn = Command(0x12, (Primitives.i8, Primitives.i8))
        self._get_data = Command(0x11, (Primitives.u8,))
        self._go_dist = Command(0x13, (Primitives.i32, Primitives.i32))

        self._manipulator_control = Command(0x20, (Primitives.u8, Primitives.u8))

        self._handshake_command = Command(0x14, (Primitives.u8,))
        self._handshake_response = b"ARDUINO_OK"

    def send_heartbeat(self) -> int:
        """heartbeat"""

        self.serial.write(self._heartbeat.pack())
        self.serial.flush()

        time.sleep(0.1)

        size = Primitives.u32.size()
        ret = self.serial.read(size)
        return Primitives.u32.unpack(ret)

    def manipulator_control(self, arm: Optional[float], claw: Optional[float]) -> None:
        """
        Управление манипулятором
        :param arm: Значение манипулятора [0..1] (None - Ось будет отключена)
        :param claw: Значение клешни [0..1] (None - Ось будет отключена)
        """

        def _norm(__v: Optional[float], _min: int, _max: int) -> int:
            if __v is None:
                return 0xFF

            __v = min(1.0, max(0.0, __v))

            return int(__v * (_max - _min)) + _min

        self.serial.write(self._manipulator_control.pack(
            _norm(arm, 90, 180),
            _norm(claw, 0, 180)
        ))

    # Обёртки над командами ниже, чтобы сразу компилировать и отправлять их в порт
    def setSpeeds(self, linear: float, angular: float) -> None:
        print(f"{linear=} {angular=}")
        self.serial.write(self._set_velocity.pack(linear, angular))

    def turn_robot(self, angle: int, speed: int) -> bool:
        self.serial.write(self._turn.pack(angle, speed))
        response = self.serial.read()
        return Primitives.u8.unpack(response)

    def go_dist(self, dist: int, speed: int) -> bool:
        self.serial.write(self._go_dist.pack(dist, speed))
        print(self._go_dist.pack(dist, speed))
        response = self.serial.read()
        print(response)
        return Primitives.u8.unpack(response)

    def get_data(self):
        self.serial.write(self._get_data.pack(1))
        data_bytes = self.serial.read(Data.data_packer.size)
        return Data.make(data_bytes)

    def is_arduino(self, timeout=0.5) -> bool:
        start_time = time.time()
        try:
            self.serial.reset_input_buffer()
            self.serial.write(self._handshake_command.pack(0xFF))
            self.serial.flush()

            while time.time() - start_time < timeout:
                if self.serial.in_waiting >= len(self._handshake_response):
                    response = self.serial.read(len(self._handshake_response))
                    return response == self._handshake_response
                time.sleep(0.01)

            return False
        except Exception:
            return False

    def close(self):
        self.serial.close()


def _launch():
    port_name = find_arduino_port()

    if port_name is None:
        print("No port!")
        return

    print(f"Find: {port_name=}")

    arduino = ArduinoConnection(Serial(port=port_name, baudrate=115200))
    time.sleep(2)

    print(arduino.send_heartbeat())

    # arduino.setSpeeds(0.5, 0.0)
    # time.sleep(1)
    #
    # arduino.setSpeeds(-0.5, 0.0)
    # time.sleep(1)

    # arduino.setSpeeds(0.0, 10)
    # time.sleep(1)
    #
    # arduino.setSpeeds(0.0, -10)
    # time.sleep(1)
    #
    # arduino.setSpeeds(0.0, 0.0)
    # time.sleep(1)

    arduino.manipulator_control(0.0, 0.0)
    time.sleep(2)

    arduino.manipulator_control(1.0, 1.0)
    time.sleep(1)

    arduino.manipulator_control(None, None)

    print(arduino.send_heartbeat())

    arduino.close()


if __name__ == '__main__':
    _launch()
