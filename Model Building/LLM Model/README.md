## Overview
README ini akan menjelaskan pembangunan model LLM dari HuggingFace yaitu BERT. Sebelum itu untuk memulai notebook tersebut disarankan untuk menggunakan Google Colab dengan cara:
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
Kami menggunakan Transformer untuk melakukan fine tuning model BERT nya dan menggunakan fuction yang ada pada libary sklearn dan tensorflow/keras untuk memprocesing data jsonnya

- Pembangunan Model <br />
Pada tahap ini saya akan menjelaskan sedikit bagaimana model ini dibuat, untuk pemahaman lebih lanjut disarankan untuk memulai notebook yang ada di repository ini.
Dimulai dari menginstall LLM BERT dan tokenizer dari Transformer tersebut
```
model_name = "bert-base-uncased"
max_len = 1024

tokenizer = BertTokenizer.from_pretrained(model_name,
                                          max_length=max_len)

model = BertForSequenceClassification.from_pretrained(model_name,
                                                      num_labels=num_labels,
                                                      id2label=id2label,
                                                      label2id = label2id)
```

Setelah beberapa proses preprocesing data kita akan membuat data train dan test untuk model
```
train_dataloader = DataLoader(train_encoding, y_train)
test_dataloader = DataLoader(test_encoding, y_test)
```

Latih model menggunakan libary dari transformer dengan argumen seperti berikut :
```
training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/Massive/output',
    do_train=True,
    do_eval=True,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    warmup_steps=10,
    weight_decay=0.05,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to="wandb"
)
```
Membuat evaluation metriks untuk mengetahui kinerja model
```
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }
```
```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=test_dataloader,
    compute_metrics= compute_metrics
)
```
```
trainer.train()
```

- Model Evaluation <br />

```
### Training and Validation Results

| Step | Training Loss | Validation Loss | Accuracy | F1      | Precision | Recall  |
|------|---------------|-----------------|----------|---------|-----------|---------|
| 50   | 4.570200      | 4.360112        | 0.034091 | 0.006034| 0.004900  | 0.029514|
| 100  | 3.924800      | 3.614377        | 0.291667 | 0.202102| 0.209414  | 0.273264|
| 150  | 3.111000      | 2.878671        | 0.545455 | 0.421424| 0.412895  | 0.513021|
| 200  | 2.371800      | 2.300424        | 0.636364 | 0.522862| 0.529903  | 0.589089|
| ...  | ...           | ...             | ...      | ...     | ...       | ...     |
| 1050 | 0.037500      | 0.386949        | 0.920455 | 0.923070| 0.929799  | 0.908431|
| 1100 | 0.036200      | 0.382586        | 0.920455 | 0.923070| 0.929975  | 0.909350|
| 1150 | 0.034900      | 0.384120        | 0.920455 | 0.923070| 0.929975  | 0.909350|
| 1200 | 0.034300      | 0.384150        | 0.920455 | 0.923070| 0.929975  | 0.909350|
| 1250 | 0.033400      | 0.384457        | 0.920455 | 0.923070| 0.929975  | 0.909350|
```
```
### Evaluation Results

|       | eval_loss | eval_Accuracy | eval_F1 | eval_Precision | eval_Recall |
|-------|-----------|---------------|---------|----------------|-------------|
| train | 0.027204  | 1.000000      | 1.00000 | 1.000000       | 1.00000     |
| test  | 0.386129  | 0.920455      | 0.90935 | 0.929975       | 0.92307     |

```

## Prototype
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.
tarok foro

## Integration
Model ini tidak diintegrasikan 

## Deployment
Model ini tidak di deploy

## Result
Model ini masih berada pada tahap model building karena chatbot yang kami gunakan untuk integrasi ke mobile adalah WatsonX Assitant
Untuk mendownload hasil dari model ini bisa dilihat pada [Drive](https://drive.google.com/drive/folders/1zfYDZhWlqwweCn97jv-DaBJHgQsQk_jX?usp=sharing) Folder chatbot.


