import socket
import struct
import cv2
import numpy as np
import time
import threading
from datetime import datetime
import json
import queue

class VideoStreamServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.client_lock = threading.Lock()
        self.running = False
        
        # Статистика
        self.stats = {
            'total_frames_received': 0,
            'start_time': time.time(),
            'clients_connected': 0
        }

        # 🔥 Очередь для хранения кадров
        self.frame_queue = queue.Queue(maxsize=1)  # Храним только последний кадр
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 🔥 Событие для ожидания новых кадров
        self.new_frame_event = threading.Event()
        
    def get_frame(self, timeout=None):
        """
        🔥 Ждет получение нового кадра и возвращает его
        Args:
            timeout: максимальное время ожидания в секундах (None - бесконечно)
        Returns:
            tuple: (frame, client_address, latency) или (None, None, None) при таймауте
        """
        # Ждем сигнала о новом кадре
        if self.new_frame_event.wait(timeout=timeout):
            self.new_frame_event.clear()  # Сбрасываем событие
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame, client_address, latency = self.latest_frame
                    return frame.copy(), client_address, latency
        return None, None, None

    def start_server(self):
        """Запуск сервера"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(0.5)  # Уменьшаем таймаут для более быстрой реакции
            
            self.running = True
            print(f"🎯 The server is running on {self.host}:{self.port}")
            print("   Waiting for clients to connect")
            
            # Запускаем потоки
            accept_thread = threading.Thread(target=self.accept_clients, daemon=True)
            stats_thread = threading.Thread(target=self.print_stats, daemon=True)
            accept_thread.start()
            stats_thread.start()
            
            # Основной цикл больше не нужен в этом методе
            # Просто ждем завершения
            accept_thread.join()
            
        except Exception as e:
            print(f"❌ Ошибка сервера: {e}")
        finally:
            self.stop_server()
    
    def accept_clients(self):
        """Прием клиентских подключений"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client_socket.settimeout(0.5)  # Добавляем таймаут для клиентского сокета
                
                with self.client_lock:
                    self.clients.append({'socket': client_socket, 'address': addr})
                    self.stats['clients_connected'] += 1
                
                print(f"✅ New client: {addr}")
                
                # Запускаем обработчик для клиента
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket, addr), 
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                continue
            except OSError as e:
                if self.running:
                    print(f"❌ Ошибка приема клиента: {e}")
                    break
            except Exception as e:
                if self.running:
                    print(f"❌ Ошибка приема клиента: {e}")
    
    def handle_client(self, client_socket, client_address):
        """Обработка клиента"""
        try:
            while self.running:
                start_time = time.time()
                
                # Получаем заголовок кадра (размер + временная метка)
                header_data = self.recv_all(client_socket, 12)
                if not header_data:
                    break
                
                # Распаковываем заголовок
                frame_size, timestamp = struct.unpack(">LQ", header_data)
                
                # Получаем данные кадра
                frame_data = self.recv_all(client_socket, frame_size)
                if not frame_data:
                    break
                
                # Декодируем JPEG
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000

                    # Обновляем статистику
                    self.stats['total_frames_received'] += 1
                    
                    # Добавляем информацию на кадр
                    self.add_overlay(frame, client_address, latency)

                    # 🔥 Сохраняем кадр и сигнализируем о новом кадре
                    with self.frame_lock:
                        self.latest_frame = (frame, client_address, latency)
                    self.new_frame_event.set()  # Сигнализируем о новом кадре
                        
        except socket.timeout:
            # Таймаут - это нормально, продолжаем цикл
            pass
        except Exception as e:
            print(f"❌ Ошибка обработки клиента {client_address}: {e}")
        finally:
            self.remove_client(client_socket)
            print(f"❌ Клиент отключен: {client_address}")
    
    def add_overlay(self, frame, client_address, latency):
        """Добавление информации на кадр"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        
        texts = [
            f"Client: {client_address}",
            f"Latency: {latency:.1f} ms",
            f"Frames: {self.stats['total_frames_received']}",
            f"Clients: {self.stats['clients_connected']}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            "Press 'q' to quit, 'c' to clear clients"
        ]
        
        for i, text in enumerate(texts):
            y_position = 30 + i * 25
            cv2.putText(frame, text, (10, y_position), font, 0.6, color, 2)
    
    def clear_clients(self):
        """Очистка отключенных клиентов"""
        with self.client_lock:
            initial_count = len(self.clients)
            self.clients = [client for client in self.clients if self.is_socket_alive(client['socket'])]
            self.stats['clients_connected'] = len(self.clients)
    
    def is_socket_alive(self, sock):
        """Проверка активности сокета"""
        try:
            # Простой способ проверки - отправка пустых данных
            sock.send(b'')
            return True
        except:
            return False
    
    def remove_client(self, client_socket):
        """Удаление клиента"""
        with self.client_lock:
            self.clients = [client for client in self.clients if client['socket'] != client_socket]
            self.stats['clients_connected'] = len(self.clients)
        
        try:
            client_socket.close()
        except:
            pass
    
    def recv_all(self, sock, n):
        """Получение всех данных"""
        data = b''
        start_time = time.time()
        while len(data) < n and self.running:
            try:
                # Устанавливаем короткий таймаут для неблокирующего чтения
                sock.settimeout(0.1)
                chunk = sock.recv(min(4096, n - len(data)))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                # Проверяем, не прошло ли слишком много времени
                if time.time() - start_time > 5.0:  # 5 секунд макс на весь кадр
                    return None
                continue
            except:
                return None
        return data if len(data) == n else None
    
    def print_stats(self):
        """Периодический вывод статистики"""
        while self.running:
            time.sleep(5)
            if self.stats['total_frames_received'] > 0:
                elapsed = time.time() - self.stats['start_time']
                fps = self.stats['total_frames_received'] / elapsed
                print(f"📊 Статистика: FPS={fps:.2f}, Клиенты={self.stats['clients_connected']}, Кадры={self.stats['total_frames_received']}")
    
    def stop_server(self):
        """Остановка сервера"""
        self.running = False
        
        with self.client_lock:
            for client in self.clients:
                try:
                    client['socket'].close()
                except:
                    pass
            self.clients.clear()
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        cv2.destroyAllWindows()
