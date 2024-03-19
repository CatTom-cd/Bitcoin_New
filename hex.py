#Функция преобразования из шестнадцатеричной системы в десятичную
def hex_to_decimal(hex_num):
   power = len(hex_num) - 1
   decimal = 0
   for digit in hex_num:
     if digit.isdigit(): # Проверяем, является ли элемент строки числом
     # конвертация цифры
       value = int(digit)
     else:
     # конвертация буквы
       value = ord(digit.upper()) - 55 # Перевод буквы в верхний регистр и возвращение числа из таблицы Unicod
     decimal += value * 16**power
     power -= 1
   return decimal