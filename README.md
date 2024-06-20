<h1 align="center">  This is a Machine Learning / AI Project </h1>

<p align="center"> 
Repository Massive III Bhaskara Chipta_AI Division
</p>

<div align="center">
    <!-- Your badges here -->
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
    <img src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white">
    <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
    <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB">
    <img src="https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white">
</div>

## Teams

- (Design Researcher)
- (Data Engineer)
- (Machine Learning Engineer)
- (Machine Learning Ops)


# Before We Start

### Overview

Repository ini akan memuat keseluruhan hasil dari apa saja yang telah kami lakukan. Pada README ini, saya akan menjelaskan lebih dalam mengenai WatsonX Assistant, sebuah layanan dari IBM Cloud yang memungkinkan pembuatan chatbot dan integrasinya yang lebih mudah ke aplikasi mobile.

Mengapa ada folder "model building"? Folder tersebut memuat dua hasil pembuatan model ML kami menggunakan TensorFlow dan Huggingface Transformer.

### Validation
![Screenshot 2024-06-18 091122](https://github.com/GufronAridho/Test1/assets/119670148/78b324a9-f44d-459f-807a-9478ac5ec598)

Pada Screnenshot diatas menampilkan penilaian kami memutuskan untuk menggunakan WatsonX Assistant untuk diintegrasikan ke aplikasi mobile project massive collab ini, peniliaanya berupa berikut
- Data Preperation
- Model Building
- Model Evaluation
- Integration
- Response

# Lets Start Building

## Idea Background

### 1. Theme
Tema : Siaga Gempa Bumi

### 2. Problem
Masalah : 

Gempa bumi, sebagai fenomena alam yang tidak dapat diprediksi secara pasti, terus menjadi ancaman serius bagi kehidupan manusia dan infrastruktur di Indonesia. Meskipun telah ada kemajuan dalam pemahaman gempa dan upaya mitigasi risiko, sejumlah tantangan kritis masih harus diatasi. Selain itu, pemahaman masyarakat mengenai ketahanan infrastruktur terhadap gempa masih terbilang minim. Penelitian prediksi gempa, pendidikan masyarakat dan pemulihan pasca-gempa juga merupakan masalah yang memerlukan perhatian lebih.Oleh karna itu, pernyataan awal masalah ini mendukung penelitian prediksi gempa dan memperkuat upaya pemulihan pasca-gempa untuk mengurangi dampak negatif gempa bumi pada manusia, ekonomi dan lingkungan, dan menyoroti perlunya fokus lebih lanjut dalam meningkatkan pemahaman masyarakat  terhadap ketahanan infrastruktur

### 3. Solution
Solusi : 

Aplikasi Peringatan Gempa yang kami hadirkan merupakan landasan kokoh dalam menjawab tantangan kritis terkait keamanan dan kesigapan masyarakat di hadapan potensi bahaya gempa. Kami tidak hanya berfokus pada memberikan informasi, tetapi juga memberikan kontribusi yang positif dan berdampak nyata dalam upaya penanggulangan dampak bencana. Dengan memadukan teknologi dan pemahaman mendalam terhadap kebutuhan masyarakat, fitur-fitur seperti SOS yang dapat melakukan pemanggilan darurat dengan cepat serta memberikan notifikasi peringatan gempa kepada warga di sekitar pusat gempa. Fitur riwayat juga memberikan riwayat lengkap dan bisa mengetahui daerah yang rawan akan terjadinya gempa. Sementara itu edukasi membantu masyarakat mengetahui informasi gempa melalui artikel dan video dan fitur Mari Bertanya  dengan implementasi chatbot yang mampu melayani masyarakat tanpa batasan waktu pada pertanyaan yang ia miliki dan juga Tidak lupa donasi akan sangat membantu warga yang terkena dampak terjadinya gempa. Dalam aplikasi ini membentuk solusi komprehensif yang menggabungkan pendidikan, respons darurat, serta aksesibilitas untuk menciptakan ekosistem yang aman dan siap menghadapi risiko gempa.

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
IBM WatsonX Assistant memungkinkan kita untuk mengupload data intents yang kita buat dalam format .csv untuk dijadikan actions dengan format data seperti berikut:
`<phrase>,<intent>`

Contoh 
```
Apa itu gempa bumi?,pengertian_gempa
Jelaskan gempa bumi,pengertian_gempa
Bisakah Anda menjelaskan tingkatan skala SIG BMKG?,tingkatan_skala_sig
Tingkatan skala SIG BMKG dijelaskan bagaimana?,tingkatan_skala_sig
Berikan saya definisi gempa vulkanik,gempa_vulkanik
Apa sih yang dimaksud dengan gempa vulkanik,gempa_vulkanik
```

### 2. Project FlowChart

![Watson Asisstant Project Flow](https://github.com/GufronAridho/Test1/assets/119670148/642a3d49-b943-4658-aad5-b4454624fcd3)

### 3. Algorithm

- Framework <br />
Kami menggunanakan WatsonX Assistant. Dengan versi algoritma terbaru (15-Apr-2023) Yang mana versi algoritma menggunakan foundation model baru untuk meningkatkan deteksi niat dan pencocokan tindakan di asisten, foundation model ini dilatih dengan menggunakan arsitektur transformator.

- Pembangunan Model

Akan saya jelaskan 2 cara untuk membuat Actions skills untuk chatbot ini, yaitu:
1. Upload Actions melalui csv intents
     - Pada halaman Actions utama klik ikon upload intens
     - Pilih file intens yang Anda miliki
     - Setelah di upload WatsonX akan memvalidasi data Anda dan melatih systemnya berdasarkan data tersebut
2. Membuat Actions secara manual
     - Pada halaman Actions utama klik "New Actions"
     - Pada bagian "Add example phrase:" Isi dengan Masukkan frasa yang diketik atau diucapkan pelanggan untuk memulai percakapan tentang topik tertentu. Frasa ini menentukan tugas, masalah, atau pertanyaan yang dimiliki pelanggan Anda. Semakin banyak frasa yang Anda masukkan, semakin baik asisten Anda mengenali apa yang diinginkan pelanggan.

Sekarang kita sudah memiliki Actions, waktu nya untuk menambah response atau jawaban dari Actions atau pertanyaan yang sudah kita buat. 
1. Pada halaman Actions utama pilih Actions yang sudah dibuat
2. Pada Conversation steps Anda bisa menambahkan apa yang harus assistant katanan, isi "Assistant says" dengan jawaban dari pertanyaan tersebut
3. Kita bisa mengkomplekskan jawaban dari chatbot ini dengan menambah Options pada "Define customer response" untuk disambungkan ke Actions lainnya, contoh jika kita masuk ke Actions "jenis gempa" kita bisa menyambungkannya ke Actions lain seperti "gemppa tektonik", "gempa vulkanik", "gempa buatan" dan lain lain. Cara nya adalah sebagai berikut:
     - Buat Options pada steps nya
     - Buat steps baru dan atur "Is taken" menjadi "with condition" dan sesuaikan conditions nya dengan Options yang telah dibuat
     - Scroll kebawah dan pada bagian "And Then" ubah menjadi "Goes to a subaction" dan atur sesuai dengan Actions mana yang Anda mau

Kita sudah membuat Action yang akan dipakai oleh chatbot untuk menjawab pertanyaan pengguna nantinya, jangan lupa untuk mereview chatbot yang telah dibuat dan mempublish dengan cara pergi ke tab Publish lalu klik Publish. 

- Model Evaluation <br />

## Prototype
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.
Kaasih foto

## Integration
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

## Deployment
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

## Result
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

## Conclusion
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.
