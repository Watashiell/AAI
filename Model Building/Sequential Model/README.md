## Overview
README ini akan menjelaskan pembangunan model Sequential menggunakan framework Tensorflow. Sebelum itu untuk memulai notebook tersebut disarankan untuk menggunakan Google Colab dengan cara:
1. Download file .ipynb
2. Buka Google Colab dan klik Open Colab
3. Buka tab Upload lalu cari atau tarik file .ipynb ini
   
# Lets Build

## Dataset and Algorithm

### 1. Dataset
- Data Collection <br />
Kami menemukan data pertanyaan kami dari berbagai situs yang memuat pembahasan mengenai gempa, beberapa situsnya adalah sebagai berikut: <br />
https://www.usgs.gov/programs/earthquake-hazards/faqs-category <br />
https://www.earthquakescanada.nrcan.gc.ca/info-gen/faq-en.php <br />
https://scweb.cwa.gov.tw/en-us/guidance/faq <br />
https://www.earthquakes.bgs.ac.uk/education/faqs/faq_index.html <br />
https://polarisdrt.org/100-frequently-asked-questions-about-earthquakes-and-their-answers/ <br />
https://www.bmkg.go.id/ <br />
https://id.quora.com/

- Data Cleaning <br />
Data Cleaning yang kami lakukan dibuat secara mmanual, dengan point pemilihan seperti :
> "Apakah pertanyaan tersebut relate ke wilayah Indonesia ?" <br />
> "Apakah pertanyaan tersebut berguna ?"

- Data Transformation <br />
Data daari pertanyaan dan jawaban yang telah didapat diubah menjadi format .json dengan struktur seperti berikut :
```
{
    "intents": [
        {
            "tag": "",
            "patterns": [
                ""
            ],
            "responses": [
                ""
            ],
            "context": [
                ""
            ]
        },
        {
            "tag": "",
            "patterns": [
                ""
            ],
            "responses": [
                ""
            ],
            "context": [
                ""
            ]
        }
    ]
}
```

Contoh 
```
{
    "intents": [
        {
            "tag": "intensitas",
            "patterns": [
                "Apa itu intensitas gempa bumi?",
                "Bisakah Anda menjelaskan tentang intensitas gempa?",
                "Bagaimana definisi intensitas dalam konteks gempa bumi?",
                "Apa yang dimaksud dengan intensitas gempa bumi?",
            ],
            "responses": [
                "Intensitas gempa bumi adalah ukuran kualitatif kekuatan gempa di suatu tempat yang ditentukan berdasarkan kerusakan dan hasil pengamatan efek gempa bumi di lokasi tersebut, yang dihitung berdasarkan pengamatan subjektif. Intensitas gempa bumi menunjukkan dampak yang dirasakan di permukaan bumi dan dapat berbeda dengan magnitudo gempa, yang menunjukkan kekuatan energi yang dihasilkan oleh gempa. Skala yang digunakan untuk mengukur intensitas gempa adalah Skala MMI (Modified Mercalli Intensity) yang dicetuskan oleh Giuseppe Mercalli pada tahun 1902. Namun, Indonesia memiliki skala intensitasnya sendiri yaitu Skala Intensitas Gempa Bumi (SIG) yang digunakan oleh BMKG."
            ],
            "context": [
                ""
            ]
        },
        {
            "tag": "peran_yang_bisa_dilakukan",
            "patterns": [
                "Apa yang bisa saya lakukan untuk korban pasca gempa",
                "Apa yang bisa saya lakukan untuk membantu korban pasca gempa?",
                "Bagaimana saya dapat memberikan kontribusi kepada korban gempa?",
                "Apakah ada cara bagi saya untuk membantu meringankan beban korban pasca gempa?",
            ],
            "responses": [
                "Ada beberapa peran yang bisa Anda lakukan untuk membantu korban pasca gempa. Salah satunya adalah dengan memberikan donasi kepada lembaga yang terpercaya. Donasi ini dapat digunakan untuk meringankan beban korban dalam hal material seperti perlengkapan hidup sehari-hari, tempat tinggal darurat, dan perawatan kesehatan. Selain itu, Anda juga bisa menjadi relawan untuk membantu pemerintah atau organisasi kemanusiaan dalam menangani pengungsian korban gempa. Dengan bergabung sebagai relawan, Anda dapat memberikan bantuan langsung kepada korban seperti pendistribusian bantuan, merawat anak-anak atau lansia, serta membantu dalam proses pemulihan dan rekonstruksi. Setiap kontribusi Anda, baik dalam bentuk donasi maupun menjadi relawan, sangat berarti bagi mereka yang terdampak gempa."
            ],
            "context": [
                ""
            ]
        }
    ]
}
```

### 2. Algorithm

- Framework <br />
Kami menggunakan framework dari Tensorflow untuk pembangunan model ini. Model yang digunakan adalah Sequential sebuah model yang memiliki tumpukan layers seperti Neural Networks dan pelatihan nya menggunakan SGD Optimizer

- Pembangunan Model <br />
Pada tahap ini saya akan menjelaskan sedikit bagaimana model ini dibuat, untuk pemahaman lebih lanjut disarankan untuk memulai notebook yang ada di repository ini.
Dimulai dari preprocesing data untuk melatih model ini
```
# Load intents.json
intents = json.loads(open('/content/intents.json').read())

# Initialize lists to hold words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['?', '!',',','.']

# Process each pattern in the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        # Add the tokenized pattern and its associated tag to documents
        documents.append((word_list, intent['tag']))
        # Add the tag to classes if it's not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem and lower each word and remove duplicates and ignored letters
words = [stemmer.stem(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save the words and classes to pickle files
pickle.dump(words, open('/content/words.pkl', 'wb'))
pickle.dump(classes, open('/content/classes.pkl', 'wb'))

# Initialize the training data
training = []
output_empty = [0] * len(classes)

# Create the bag of words and output row for each document
for document in documents:
    bag =[]
    word_patterns = document[0]
    word_patterns = [stemmer.stem(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train_x and train_y
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])
```

Setelah beberapa proses preprocesing data, saatnya melatih model
```
# Build the model
model = Sequential()
model.add(Input(shape=(len(train_x[0]),)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=300, batch_size=32, validation_split=0.2, verbose=1)

# Save the model
model.save('chatbotmodel.h5', hist)

print('Done')
```
- Model Evaluation <br />

```
Epoch 1/300
27/27 [==============================] - 3s 14ms/step - loss: 4.6232 - accuracy: 0.0119 - val_loss: 4.6287 - val_accuracy: 0.0142
Epoch 2/300
27/27 [==============================] - 0s 4ms/step - loss: 4.6131 - accuracy: 0.0142 - val_loss: 4.6163 - val_accuracy: 0.0237
Epoch 3/300
27/27 [==============================] - 0s 7ms/step - loss: 4.5964 - accuracy: 0.0154 - val_loss: 4.6038 - val_accuracy: 0.0284
......
Epoch 298/300
27/27 [==============================] - 0s 5ms/step - loss: 0.2600 - accuracy: 0.9253 - val_loss: 0.1943 - val_accuracy: 0.9479
Epoch 299/300
27/27 [==============================] - 0s 5ms/step - loss: 0.2393 - accuracy: 0.9407 - val_loss: 0.1922 - val_accuracy: 0.9573
Epoch 300/300
27/27 [==============================] - 0s 10ms/step - loss: 0.2447 - accuracy: 0.9276 - val_loss: 0.1926 - val_accuracy: 0.9621

```
```
33/33 [==============================] - 0s 2ms/step
Classification Report:
                                       precision    recall  f1-score   support

                               P_wave       1.00      1.00      1.00        18
                               S_wave       1.00      1.00      1.00        17
                           aftershock       1.00      1.00      1.00         9
                      alat_ukur_gempa       1.00      0.92      0.96        13
                 bangunan_tahan_gempa       1.00      1.00      1.00         5
                  bantuan_pasca_gempa       1.00      1.00      1.00         8
                             basarnas       1.00      1.00      1.00        12
               bisakah_prediksi_gempa       1.00      1.00      1.00         9
             bisakah_prediksi_tsunami       1.00      1.00      1.00         9
                                 bmkg       1.00      1.00      1.00        12
                                 bnpb       1.00      1.00      1.00        12
                                 bpbd       1.00      1.00      1.00        12
                           cincin_api       1.00      1.00      1.00         7
                         dampak_gempa       0.92      1.00      0.96        11
                            foreshock       1.00      1.00      1.00        10
                 gempa_akibat_patahan       1.00      1.00      1.00         8
                         gempa_buatan       0.94      1.00      0.97        16
                gempa_dampak_meteorit       1.00      1.00      1.00        16
                gempa_mematikan_dunia       1.00      1.00      1.00         6
                       gempa_tektonik       1.00      1.00      1.00        18
                 gempa_terbesar_dunia       1.00      1.00      1.00         7
             gempa_terbesar_indonesia       1.00      1.00      1.00         6
                       gempa_vulkanik       1.00      1.00      1.00        16
                 hewan_prediksi_gempa       1.00      1.00      1.00         6
           hiposentrum_dan_episentrum       1.00      1.00      1.00        11
                           intensitas       1.00      1.00      1.00         9
                                 inti       1.00      1.00      1.00        16
                          jenis_gempa       1.00      1.00      1.00        16
                          jenis_kerak       1.00      1.00      1.00        12
                                 kamu       1.00      1.00      1.00         7
                           kerak_bumi       1.00      1.00      1.00        17
                           lama_gempa       1.00      1.00      1.00        10
                         lapisan_bumi       1.00      1.00      1.00        19
                              lembaga       1.00      1.00      1.00         9
                              lempeng       1.00      1.00      1.00        11
                            magnitudo       1.00      1.00      1.00        10
                            mainshock       1.00      1.00      1.00        10
                               mantel       1.00      1.00      1.00        17
                       meletus_gunung       1.00      1.00      1.00         6
                             mitigasi       1.00      1.00      1.00        16
                mitigasi_dalam_gedung       1.00      1.00      1.00         9
                  mitigasi_dalam_lift       1.00      1.00      1.00         9
                 mitigasi_dalam_rumah       1.00      1.00      1.00         9
                mitigasi_luar_ruangan       1.00      1.00      1.00         9
                mitigasi_tempat_ramai       1.00      1.00      1.00         9
                          mitos_kecil       1.00      1.00      1.00         5
                      mitos_kerusakan       1.00      0.80      0.89         5
                          mitos_pintu       1.00      1.00      1.00         5
                          mitos_telan       1.00      1.00      1.00         7
                          musim_gempa       1.00      1.00      1.00         7
                             notfound       0.78      1.00      0.88         7
                              patahan       1.00      1.00      1.00        18
                     patahan_divergen       1.00      1.00      1.00        13
                        patahan_geser       1.00      1.00      1.00        13
                       patahan_miring       1.00      1.00      1.00        13
                         patahan_naik       1.00      1.00      1.00        13
                       patahan_normal       1.00      1.00      1.00        13
                     patahan_terbalik       1.00      1.00      1.00        13
                            pelatihan       1.00      1.00      1.00         5
                    pemerintah daerah       1.00      1.00      1.00         7
          penanganan_kesehatan_mental       1.00      1.00      1.00        13
                     pengalaman_gempa       1.00      1.00      1.00         5
                      pengaruh_mental       1.00      1.00      1.00        15
                     pengertian_gempa       1.00      1.00      1.00        15
                          pengungsian       1.00      1.00      1.00         5
                       penyebab_gempa       1.00      0.93      0.96        14
                   penyelamatan_hewan       1.00      1.00      1.00         4
            peran_yang_bisa_dilakukan       1.00      1.00      1.00         7
       perbedaan_intensitas_magnitudo       0.93      1.00      0.96        13
perbedaan_skala_ritcher_dan_magnitudo       1.00      1.00      1.00         9
                            persiapan       1.00      1.00      1.00         8
                                  pmi       1.00      1.00      1.00        12
                        purnama_gempa       1.00      1.00      1.00         6
                              relawan       0.88      1.00      0.93         7
                      rencana_tanggap       1.00      1.00      1.00         8
                              richter       1.00      1.00      1.00        12
                               sapaan       1.00      0.95      0.97        20
                                saran       1.00      1.00      1.00        13
                              seismik       1.00      1.00      1.00        20
                           seismologi       1.00      1.00      1.00         9
              sektor_agrikultur_gempa       1.00      0.89      0.94         9
                 sektor_ekonomi_gempa       1.00      1.00      1.00         6
              sektor_pendidikan_gempa       1.00      1.00      1.00         7
                  sektor_sosial_gempa       1.00      1.00      1.00         6
                        setelah_gempa       1.00      1.00      1.00         7
                         siap_dokumen       1.00      1.00      1.00         8
                       simulasi_gempa       1.00      1.00      1.00         5
               sistem_peringatan_dini       1.00      1.00      1.00         8
                          skala_gempa       0.91      1.00      0.95        10
                            skala_mmi       1.00      1.00      1.00         8
                            skala_sig       0.89      1.00      0.94         8
                   tempat_paling_aman       1.00      1.00      1.00         6
                         terima_kasih       1.00      0.94      0.97        18
            tingkatan_skala_magnitudo       1.00      1.00      1.00         8
                  tingkatan_skala_mmi       1.00      1.00      1.00        10
                  tingkatan_skala_sig       1.00      0.90      0.95        10
                           tipe_gempa       1.00      1.00      1.00        13
                              tsunami       1.00      1.00      1.00        19
                 tsunami_akibat_gempa       1.00      1.00      1.00         7
               tsunami_terparah_dunia       1.00      1.00      1.00         6
           tsunami_terparah_indonesia       1.00      1.00      1.00         5
                  wilayah_rawan_gempa       1.00      0.86      0.92         7

                             accuracy                           0.99      1054
                            macro avg       0.99      0.99      0.99      1054
                         weighted avg       0.99      0.99      0.99      1054

```

## Prototype
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.
tarok foro

## Integration
Model ini belum bisa diintegrasikan ke aplikasi mobile, tetapi ML Eng kami mampu untuk mengitegrasikan model ini react js pada folder "my-chatbot", dengan kekurangan yaitu belum ditegrasikan ke website utama project collab ini dan deployment app flask nya masih dilakukan secara lokal.

## Deployment
Model ini menggunakan flask CORS, exios untuk memungkinkannya berkomunikasi antara front end dan back end website. Tetapi seperti yang dibilang sebelumnya kami masih belum bisa untuk mendeploy model ini ke internet

## Result
Model ini masih berada sudah melewati tahap model building, akan tetapi masih belum mampu untuk menyelesaikan tahap deployment dan integration.

kasih foto

## Conclusion



