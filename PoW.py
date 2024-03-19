 # Реализация алгоритма PoW
import hashlib
import datetime
import time
 
# ХЭШ-ЗНАЧЕНИЕ ВСЕХ ТРАНЗАКЦИЙ БЛОКА, ПОЛУЧЕННЫХ ПО АЛГОРИТМУ 'ДЕРЕВО МЕРКЛЕ'
# Функция хеширования данных
def hash_data(data, hash_function = 'sha256'):
  hash_function = getattr(hashlib, hash_function) # Извлечение значения из атрибута по имени
  data = data.encode('utf-8') # Кодирование данных
  return hash_function(data).hexdigest() # Процесс хеширования данных
 
def hash_list(lst, hash_function = 'sha256'):
  lst_new = []
  for i in range(lst):
    transaction = 'name' + str(i)
    sha = hashlib.sha256(str(transaction).encode('utf-8')).hexdigest()# Хешируем транзакции
    lst_new.append(sha)  
  # print(lst_new)
 
  assert len(lst_new) > 2, 'There are no transactions' # Если условие ложно, то оператор останавливает выполнение программы
  n = 0
  while len(lst_new) > 1:
    if len(lst_new) % 2 == 0: # Если количество транзакций четное
      mass = []
      while len(lst_new) > 1:
        a = lst_new.pop(0) # Получение элемента по списку и его удаление
        b = lst_new.pop(0)
        mass.append(hash_data(a + b, hash_function))
      lst_new = mass
    else: # Если количество транзакций нечётное
      mass = []
      der = lst_new.pop(-1) # Дублирование последней транзакции
      while len(lst_new) > 1:
        a = lst_new.pop(0) # Получение элемента по списку и его удаление
        b = lst_new.pop(0)
        mass.append(hash_data(a + b, hash_function))
      mass.append(der)
      lst_new = mass
  return lst_new
 
# Указываем количество транзакций, находим Хэш транзакции и формируем из них список хэш-значений
number_of_transactions = int(input())
data = hash_list(number_of_transactions)
print('Хэш-значение блока данных', data)
 
#---------------------------------------------------------------------------------------------
# Временная метка
timestamp = datetime.datetime.now()
 
#---------------------------------------------------------------------------------------------
first_block_header = hashlib.sha256(str(timestamp).encode('utf-8') + str(data).encode('utf-8')).hexdigest()
print('Хэш-значение заголовка первичного блока', first_block_header)
 
#---------------------------------------------------------------------------------------------
max_nonce = 2**32
def proof_of_work(header, difficulty_bits):
  # Вычисляем сложность достижения цели
  target = 2**(256-difficulty_bits)
  # Вычисляем хэш-значение блока
  for nonce in range(max_nonce):
    hash_result_now_block = hashlib.sha256(str(header).encode('utf-8') + str(nonce).encode('utf-8')).hexdigest()
 
    # Производим проверку хэш-значения. Необходимо, чтобы полученый результат был меньше целевого значения(target)
    if int(hash_result_now_block, 16) < target:
      print(f'Success, nonce is {nonce}')
      print(f'Hash is {hash_result_now_block}')
      return hash_result_now_block, nonce
  return nonce
 
# Проверим как будет изменяться время поиска решения при изменении difficulty_bits 
if __name__ == '__main__':
  nonce = 0
  hash_result_now_block = ''
 
  # Определяем уровень сложности от 0 до 32 бита
  for difficulty_bits in range(32):
    diffuculty = 2**difficulty_bits
    print('Difficulty:', diffuculty, 'bits:', difficulty_bits)
 
    # Определение текущего времени
    start_time = time.time()
    # Создание нового блока, включающего хэш-значение предыдущего
    new_block = str(first_block_header) + hash_result_now_block
    # Поиск значения nonce  для нового блока
    hash_result_now_block, nonce = proof_of_work(new_block, difficulty_bits)
    # Фиксация времени, которое потребовалось для нахождения значения
    end_time = time.time()
 
    result_time = end_time - start_time
    #
    print('Elapsed time: ', result_time, 'seconds')
 
    # Увеличение уровня сложности на 1 бит приводит к увеличению времени, затраченного на поиск решения