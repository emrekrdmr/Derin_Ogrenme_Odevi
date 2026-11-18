"""Derin Öğrenme ve CNN Ödevi
Hazırlayan: Emre Karademir
Öğrenci No: 02230224075
Gerekli kütüphaneler: tensorflow, numpy, matplotlib
Kurulum: python -m pip install tensorflow-cpu==2.20.0 numpy matplotlib
Çalıştırma: python cnn_odevi.py
"""

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


klasor = Path(__file__).resolve().parent / "sonuclar"


def konvolusyon_ornegi():
    tf.keras.utils.set_random_seed(42)
    katman = tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation="relu",
        kernel_initializer="he_normal"
    )
    giris = tf.random.normal((1, 128, 128, 1))
    cikis = katman(giris)
    agirliklar, biaslar = katman.get_weights()

    print("\n3. Konvolüsyon")
    print("Giriş boyutu:", giris.shape)
    print("Çıkış boyutu:", cikis.shape)
    print("Ağırlık boyutu:", agirliklar.shape)
    print("Bias boyutu:", biaslar.shape)
    print("İlk filtre:\n", agirliklar[:, :, 0, 0])

    fig, eksenler = plt.subplots(4, 4, figsize=(7, 6))
    sinir = np.abs(agirliklar).max()
    for i, eksen in enumerate(eksenler.flat):
        eksen.imshow(agirliklar[:, :, 0, i], cmap="gray", vmin=-sinir, vmax=sinir)
        eksen.set_title(f"Filtre {i + 1}", fontsize=9)
        eksen.axis("off")
    plt.tight_layout()
    fig.savefig(klasor / "filtreler.png", dpi=180)
    plt.close(fig)


def pooling_ve_aktivasyon():
    giris = tf.constant([
        [1., 3., 2., 4.], [5., 6., 1., 2.],
        [7., 2., 8., 3.], [4., 1., 5., 9.]
    ])
    giris = tf.reshape(giris, (1, 4, 4, 1))
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    cikis = pooling(giris)
    print("\n4. Max Pooling")
    print("Giriş:\n", giris.numpy()[0, :, :, 0])
    print("Çıkış:\n", cikis.numpy()[0, :, :, 0])
    print("Boyutlar:", giris.shape, cikis.shape)

    x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
    print("\n5. Aktivasyon fonksiyonları")
    print("ReLU giriş:", x.numpy())
    print("ReLU çıkış:", tf.keras.layers.ReLU()(x).numpy())
    puanlar = tf.constant([[2., 1., 0.1]])
    olasiliklar = tf.keras.layers.Softmax()(puanlar)
    print("Softmax:", olasiliklar.numpy())
    print("Toplam:", tf.reduce_sum(olasiliklar).numpy())
    print("Tahmin edilen sınıf:", int(tf.argmax(olasiliklar[0])))


def model_olustur(filtre_sayisi):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filtre_sayisi, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def grafik_ciz(gecmis, filtre_sayisi):
    fig, eksenler = plt.subplots(1, 2, figsize=(10, 3.5))
    turlar = range(1, 6)
    for eksen, olcut, baslik in zip(eksenler, ["accuracy", "loss"], ["Doğruluk", "Kayıp"]):
        eksen.plot(turlar, gecmis.history[olcut], "o-", color="black", label="Eğitim")
        eksen.plot(turlar, gecmis.history["val_" + olcut], "s--", color="gray", label="Doğrulama")
        eksen.set_title(f"{filtre_sayisi} filtre - {baslik}")
        eksen.set_xlabel("Epoch")
        eksen.set_xticks(list(turlar))
        eksen.legend()
    plt.tight_layout()
    fig.savefig(klasor / f"egitim_{filtre_sayisi}.png", dpi=180)
    plt.close(fig)


def main():
    klasor.mkdir(exist_ok=True)
    konvolusyon_ornegi()
    pooling_ve_aktivasyon()

    print("\n6. Fashion-MNIST")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    print("Eğitim verisi:", x_train.shape)
    print("Test verisi:", x_test.shape)

    sonuclar = []
    gecmisler = {}
    for filtre in [8, 16]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = model_olustur(filtre)
        print(f"\nİlk katmandaki filtre sayısı: {filtre}")
        model.summary()
        baslangic = time.perf_counter()
        gecmis = model.fit(x_train, y_train, epochs=5, batch_size=64,
                           validation_split=0.1, verbose=2)
        sure = time.perf_counter() - baslangic
        test_kaybi, test_dogrulugu = model.evaluate(x_test, y_test, verbose=0)
        sonuclar.append({
            "filtre": filtre,
            "egitim_dogrulugu": gecmis.history["accuracy"][-1],
            "test_dogrulugu": test_dogrulugu,
            "test_kaybi": test_kaybi,
            "sure_saniye": sure
        })
        gecmisler[str(filtre)] = gecmis.history
        grafik_ciz(gecmis, filtre)

    print("\n8. Deney karşılaştırması")
    for sonuc in sonuclar:
        print(f"{sonuc['filtre']} filtre: Eğitim %{100 * sonuc['egitim_dogrulugu']:.2f}, "
              f"Test %{100 * sonuc['test_dogrulugu']:.2f}, "
              f"Kayıp {sonuc['test_kaybi']:.4f}, Süre {sonuc['sure_saniye']:.2f} sn")
    with open(klasor / "deney_sonuclari.csv", "w", encoding="utf-8-sig", newline="") as dosya:
        yazici = csv.DictWriter(dosya, fieldnames=list(sonuclar[0]))
        yazici.writeheader()
        yazici.writerows(sonuclar)
    with open(klasor / "egitim_kaydi.json", "w", encoding="utf-8") as dosya:
        json.dump({"sonuclar": sonuclar, "gecmisler": gecmisler,
                   "tensorflow": tf.__version__}, dosya, ensure_ascii=False, indent=2)
    print("Grafikler ve deney sonuçları kaydedildi:", klasor)


if __name__ == "__main__":
    main()
