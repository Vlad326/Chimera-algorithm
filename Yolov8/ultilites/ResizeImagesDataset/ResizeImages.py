from PIL import Image
import os

# Пути к папкам
input_folder = 'train_resized_images'
output_folder = 'train_resized_images'

# Создаем выходную папку, если ее нет
os.makedirs(output_folder, exist_ok=True)

# Новый размер
new_size = (1280, 733)

# Обрабатываем все изображения в папке
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            # Открываем изображение
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Меняем размер
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Сохраняем
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)
            
            print(f'Обработано: {filename}')
        except Exception as e:
            print(f'Ошибка при обработке {filename}: {e}')

print('Все изображения обработаны!')
