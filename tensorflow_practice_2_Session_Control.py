# This practice is for practicing the use of Session
# Assume we want to get the result of matrix 1 multiply matrix2


import tensorflow as tf

# Create a matrix with size 1*2
matrix1 = tf.constant([[3,3]])
# Create a matrix with size 2*1
matrix2 = tf.constant([[2],[2]])

# 用來表示相乘這件任務
# If we use numpy, this line is as same as "np.dot(matrix1,matrix2)"
product = tf.matmul(matrix1,matrix2)

# Use method 1 of Session 
# result 是執行過後的結果
sess = tf.Session()
result = sess.run(product)
print("Method1: ",result)
sess.close()

# Use method 2 of Session 
# result2 是執行過後的結果
# In this type of use , Session結束後會自動關上!
with tf.Session() as sess:
    result2 = sess.run(product)
    print("Method2: ",result2)
