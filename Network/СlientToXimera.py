import json
import hashlib
from datetime import datetime
import numpy as np
import socket


def generate_checksum(data):
    def custom_serializer(obj):
        if isinstance(obj, np.float32):
            return round(float(obj), 6)  # Округление до 6 знаков
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    json_str = json.dumps(data, default=custom_serializer, separators=(',', ':'))
    return hashlib.md5(json_str.encode()).hexdigest()


class SenderToXimera:
    def __init__(self):
        with open("C:/Users/User/PycharmProjects/PythonProject/Project/algoritmXimeraV1/Network/ConnectionToXimera.json", "r") as f:
            self.connection_data = json.load(f)

    def SendToXimeraData(self, port=5000, InputData={}):
        # 1 ОШИБКА В JSON ФАЙЛЕ НЕ ДОЛЖНО БЫТЬ ПРОБЕЛОВ И ВСЕ С ОДНУ СТРОЧКУ
        # 2 ОШИБКА В JSON ФАЙЛЕ В ДАННЫХ XIMERA ДОЛЖНО БЫТЬ ОКРУГЛЕНА (НА ESP 32 МАКС 6 ЦИФР ПОСЛЕ ЗАПЯТОЙ!)
        # ВСЕ ОШИБКИ ИСПРАВЛЕНЫ!

        #msg = {"timestamp":"2025-08-04T01:19:40.283","input":{"Ximera":{"X":-1,"Y":-1,"R":-1},"Opponent":{"X":1091.948,"Y":665.8186,"R":189.46713}}}
        msg = {"connection": {"server_ip": self.connection_data["server_ip"],"client_ip": self.connection_data["client_ip"]}, "data":{"timestamp": datetime.now().isoformat(timespec="milliseconds"), "input":InputData}}
        #datetime.now().isoformat(timespec="milliseconds")

        dataMSG = msg["data"]
        
        """if type(dataMSG["input"]["Ximera"]["X"]) != int:
            dataMSG["input"]["Ximera"]["X"] = round(float(dataMSG["input"]["Ximera"]["X"]), 6)
            dataMSG["input"]["Ximera"]["Y"] = round(float(dataMSG["input"]["Ximera"]["Y"]), 6)"""

        dataMSG["input"]["Ximera"]["X"] = int(dataMSG["input"]["Ximera"]["X"])
        dataMSG["input"]["Ximera"]["Y"] = int(dataMSG["input"]["Ximera"]["Y"])
        dataMSG["input"]["Ximera"]["Side"] = int(dataMSG["input"]["Ximera"]["Side"])

        if type(dataMSG["input"]["Ximera"]["R"]) != int:
            dataMSG["input"]["Ximera"]["R"] = round(float(dataMSG["input"]["Ximera"]["R"]), 6)

        dataMSG["input"]["Opponent"]["X"] = int(dataMSG["input"]["Opponent"]["X"])
        dataMSG["input"]["Opponent"]["Y"] = int(dataMSG["input"]["Opponent"]["Y"])

        if type(dataMSG["input"]["Opponent"]["R"]) != int:
            dataMSG["input"]["Opponent"]["R"] = round(float(dataMSG["input"]["Opponent"]["R"]), 6)

        # Генерируем контрольную сумму ДО отправки
        msg["checksum"] = generate_checksum(msg)

        
        #print(msg)
        # Сериализация с округлением и компактным форматом
        message_str = json.dumps(msg, default=lambda o: round(float(o), 6)
        if isinstance(o, np.float32) else None,
                                 separators=(',', ':'))
 
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(message_str.encode(), (self.connection_data["server_ip"], port))
        #print(f"Send data to ESP32 {message_str}")
