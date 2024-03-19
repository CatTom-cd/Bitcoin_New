import hashlib
transactions = []
def now_hash(num):
  for i in range(num):
    transaction = 'name' + str(i)
    sha = hashlib.sha256(str(transaction).encode('utf-8')).hexdigest()
    transactions.append(sha)
    # print(transactions)
  return transactions