#include <WiFi.h>
#include <ArduinoJson.h>
#include <MD5Builder.h>

const char* ssid = "SSID";
const char* password = "PASSWORD";
WiFiUDP udp;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  Serial.print("Подключение к WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nПодключено! IP адрес: " + WiFi.localIP().toString());

  udp.begin(5000);
  Serial.println("ESP32 готов к приёму данных на порту 5000");
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char packet[2048];
    int len = udp.read(packet, sizeof(packet) - 1); // Оставляем место для нуль-терминатора
    if (len > 0) {
      packet[len] = '\0';

      // Проверяем размер пакета
      if (len >= sizeof(packet)) {
        Serial.println("Ошибка: пакет слишком большой!");
        return;
      }

      DynamicJsonDocument doc(2048);
      DeserializationError error = deserializeJson(doc, packet);

      if (error) {
        Serial.print("Ошибка десериализации JSON: ");
        Serial.println(error.c_str());
        return;
      }

      // Проверяем наличие всех обязательных полей
      if (!doc.containsKey("checksum") || !doc.containsKey("data")) {
        Serial.println("Ошибка: неверный формат пакета (отсутствуют обязательные поля)");
        return;
      }

      // Проверяем контрольную сумму
      String received_checksum = doc["checksum"].as<String>();

      // Создаем копию документа без checksum для проверки
      DynamicJsonDocument docForChecksum(2048);
      docForChecksum.set(doc);
      docForChecksum.remove("checksum");

      String jsonStr;
      serializeJson(docForChecksum, jsonStr);

      MD5Builder md5;
      md5.begin();
      md5.add(jsonStr);
      md5.calculate();
      String computed_checksum = md5.toString();

      if (computed_checksum.equals(received_checksum)) {
        Serial.println("\nДанные валидны:");

        // Извлекаем данные подключения
        //JsonObject conn = doc["connection"];
        //const char* server_ip = conn["server_ip"];
        //const char* client_ip = conn["client_ip"];

        // Извлекаем полезные данные
        JsonObject data = doc["data"];
        const char* timestamp = data["timestamp"];
        JsonObject input = data["input"];

        // Выводим информацию о подключении
        //Serial.printf("Сервер: %s, Клиент: %s\n", server_ip, client_ip);
        Serial.printf("Временная метка: %s\n", timestamp);

        // Выводим данные роботов
        if (input.containsKey("Ximera")) {
          JsonObject ximera = input["Ximera"];
          float XimeraX = ximera["X"];
          float XimeraY = ximera["Y"];
          float XimeraR = ximera["R"];
          Serial.printf("Ximera - X: %.2f, Y: %.2f, R: %.2f\n", XimeraX, XimeraY, XimeraR);
        }

        if (input.containsKey("Opponent")) {
          JsonObject opponent = input["Opponent"];
          float OpponentX = opponent["X"];
          float OpponentY = opponent["Y"];
          float OpponentR = opponent["R"];
          Serial.printf("Opponent - X: %.2f, Y: %.2f, R: %.2f\n", OpponentX, OpponentY, OpponentR);
        }

        //



      } else {
        Serial.println("Ошибка: контрольная сумма не совпадает!");
        Serial.println("Ожидалось: " + computed_checksum);
        Serial.println("Получено: " + received_checksum);
      }
    }
  }
}