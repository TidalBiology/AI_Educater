try:
    import tensorflow as tf
    import matplotlib.pyplot as plt
    print("Gerekli kütüphaneler yüklü!")
except ImportError as e:
    print(f"Kütüphane eksik: {e}")

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Veri setini yükleyin ve eğitim/test ayrımı yapın
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veriyi normalize edin (0-255 aralığını 0-1 arasına dönüştürme)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Veri boyutlarını yazdırın
print(f"Eğitim verisi boyutu: {x_train.shape}")
print(f"Test verisi boyutu: {x_test.shape}")

# Modeli oluşturun
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Giriş katmanı: 28x28 görüntüyü düzleştirir
    Dense(256, activation='relu'),  # 256 nöronlu gizli katman
    Dropout(0.2),  # Aşırı öğrenmeyi önlemek için dropout
    Dense(128, activation='relu'),  # 128 nöronlu gizli katman
    Dropout(0.2),
    Dense(10, activation='softmax')  # Çıkış katmanı: 10 sınıf
])

# Modelin özetini yazdırın
model.summary()

# Modeli derleyin
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitin
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=64)

# Modeli test edin
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest doğruluğu: {test_acc:.2f}")

# Eğitim ve doğrulama doğruluğunu görselleştirin
plt.figure(figsize=(12, 5))

# Doğruluk
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()

plt.show()
