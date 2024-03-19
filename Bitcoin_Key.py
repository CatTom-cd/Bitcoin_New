import bitcoin

#Генерация случайного приватного ключа
valid_private_key = False
while not valid_private_key:
    private_key = bitcoin.random_key()
    decoder_private_key = bitcoin.decode_privkey(private_key, 'hex')
    valid_private_key =0 < decoder_private_key < bitcoin.N
print('Private Key (hex)  is : ', private_key)
print('Private Key (decimak) is : ', decoder_private_key)
print('------------------------------------------------------------------')

#Преобразование секретного ключа в формат WIN
wif_encoder_private_key = bitcoin.encode_privkey(decoder_private_key, 'wif')
print('Private Key Compressed (hex) is: ', wif_encoder_private_key)
print('------------------------------------------------------------------')

# Сжимаем секретный ключ путем добавления суффикса 01
compressed_private_key = private_key + '01'
print('Private key Compressed (hex) is:', compressed_private_key)
print('------------------------------------------------------------------')

# Генерация формата WIF из сжатого секретного ключа
wif_compressd_private_key = bitcoin.encode_privkey(bitcoin.decode_privkey(compressed_private_key, 'hex'), 'wif')
print('Private key Compressed (wif - compressed) is:', wif_compressd_private_key)
print('------------------------------------------------------------------')

# Умножение базовой точки генерации G на эллептической кривой на секретный ключ для получения точки открытого ключа
public_key = bitcoin.fast_multiply(bitcoin.G, decoder_private_key)
print('Prublic key (x, y) coordinates is:', public_key)
print('------------------------------------------------------------------')

#Кодирование в шестнадцатиричном формате с префиксом 04
hex_encoded_public_key = bitcoin.encode_pubkey(public_key, 'hex')
print('Public key (hex) is:', hex_encoded_public_key)
print('------------------------------------------------------------------')

#Сжатие открытого ключа, выбор префикса в зависимости от четности или нечетности координаты Y
(public_key_x, public_key_y) = public_key
if (public_key_y % 2) == 0:
    compressed_prefix = '02'
else:
    compressed_prefix = '03'
hex_compressed_public_key = compressed_prefix + bitcoin.encode(public_key_x, 16)
print('Compressed Public Key (hex) is: ', hex_compressed_public_key)
print('------------------------------------------------------------------')

#Генерация биткойн-адресcа из открытого ключа
print('Bitcoin address (b58check) is: ', bitcoin.pubkey_to_address(public_key))
print('------------------------------------------------------------------')

#Генерация сжатого биткойн-адресcа из сжатого открытого ключа
print('Compressed Bitcoin Address (b58check) is:', bitcoin.pubkey_to_address(hex_compressed_public_key))
