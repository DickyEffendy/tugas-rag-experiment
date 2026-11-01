# RAG Parameter Exploration

## 1. Informasi Eksperimen

| Komponen        | Keterangan                                                                                                        |
|:----------------|:------------------------------------------------------------------------------------------------------------------|
| Nama PDF        | `Artikel_PM25_UNAMA.pdf`                                                                                          |
| Jumlah halaman  | 8 halaman                                                                                                         |
| Topik           | Prediksi konsentrasi PM2,5 di Kota Jambi dan perbandingan Regresi Linier Berganda dengan Random Forest Regression |
| Aplikasi        | Simple RAG                                                                                                        |
| Vector database | Qdrant                                                                                                            |
| Model embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`                                                     |
| Model LLM       | `openai/gpt-oss-120b` melalui Groq                                                                                |

Satu PDF dan dua pertanyaan yang sama digunakan pada seluruh eksperimen.
Setiap eksperimen mengubah satu parameter, sementara parameter lainnya
dipertahankan.

## 2. Pertanyaan Pengujian

**Q1.** Apa saja variabel independen dan variabel dependen yang
digunakan dalam penelitian untuk memprediksi konsentrasi PM2,5 di Kota
Jambi?

**Q2.** Berapa nilai R², RMSE, MAE, MAPE, dan akurasi untuk
masing-masing algoritma Regresi Linier Berganda dan Random Forest
Regression berdasarkan Tabel 2 Hasil Pengujian pada halaman 6?

## 3. Baseline Configuration

| Parameter       |                                                         Nilai |
|:----------------|--------------------------------------------------------------:|
| Temperature     |                                                           0.2 |
| TOP_K           |                                                             4 |
| CHUNK_SIZE      |                                                           100 |
| CHUNK_OVERLAP   |                                                            20 |
| Jumlah chunks   |                                                            85 |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| LLM model       |                                         `openai/gpt-oss-120b` |

Baseline menghasilkan jawaban Q1 dan Q2 secara lengkap.

------------------------------------------------------------------------

## 4. Experiment 1: Temperature

### Temperature = 0.0

**Actual Output Q1:** Variabel independen yang digunakan adalah suhu
udara, kelembapan udara, kecepatan angin, curah hujan, dan jumlah titik
panas. Variabel dependennya adalah konsentrasi PM2,5. Sumber: halaman 2.

**Actual Output Q2:** Output yang tercatat hanya menyampaikan bahwa
Random Forest Regression memiliki RMSE lebih rendah dan dinilai lebih
baik daripada Regresi Linier Berganda. Seluruh angka R², RMSE, MAE,
MAPE, dan akurasi tidak muncul secara lengkap. Source dan similarity
score Q2 tidak tercatat lengkap.

**Observation:** Jawaban Q1 tetap memuat lima parameter cuaca dan
konsentrasi PM2,5. Q2 belum menyajikan seluruh metrik. Temperature tidak
mengubah retrieval, tetapi keluaran generation pada pengujian ini belum
memberikan seluruh rincian yang diminta.

### Temperature = 0.2

**Actual Output Q1:** Variabel independen adalah suhu udara, kelembapan
udara, kecepatan angin, curah hujan, dan jumlah titik panas. Variabel
dependen adalah konsentrasi PM2,5. Sumber: halaman 2.

**Actual Output Q2:**

| Metrik       | Regresi Linier Berganda | Random Forest Regression |
|:-------------|------------------------:|-------------------------:|
| R²           |                   0,585 |                    0,588 |
| RMSE (µg/m³) |                   6,904 |                    6,756 |
| MAE (µg/m³)  |                   5,672 |                    5,490 |
| MAPE (%)     |                    26,1 |                     26,0 |
| Akurasi (%)  |                    73,9 |                     74,0 |

Sumber: halaman 7, 2, 6, dan 5. Tabel 2 sebagai sumber utama berada pada
halaman 6.

**Similarity score baseline yang tercatat:**

| Pertanyaan | Halaman | Similarity score |
|:-----------|--------:|-----------------:|
| Q1         |       2 |         0,882426 |
| Q1         |       1 |         0,830789 |
| Q2         |       7 |         0,822853 |
| Q2         |       2 |         0,816785 |
| Q2         |       6 |         0,807034 |
| Q2         |       5 |         0,786520 |

**Observation:** Temperature 0.2 menghasilkan jawaban Q1 dan Q2 secara
lengkap pada baseline. Random Forest Regression memiliki R² dan akurasi
sedikit lebih tinggi, serta RMSE, MAE, dan MAPE lebih rendah.

### Temperature = 0.8

**Actual Output Q1:** Jawaban tetap menyebutkan suhu udara, kelembapan
udara, kecepatan angin, curah hujan, dan jumlah titik panas sebagai
variabel independen. Variabel dependen tetap konsentrasi PM2,5.

**Actual Output Q2:** Catatan pengujian menunjukkan keluaran yang tidak
sepenuhnya konsisten. Salah satu keluaran memuat tabel metrik lengkap
yang sama dengan baseline 0.2. Pada pengulangan lain, jawaban hanya
menyebutkan bahwa Random Forest Regression memiliki RMSE lebih rendah
dan akurasi lebih tinggi, tanpa seluruh angka. Pada keluaran lengkap,
sumber yang tercatat adalah halaman 7, 2, 6, dan 5. Similarity score
khusus temperature 0.8 tidak tercatat.

**Observation:** Isi faktual Q1 tidak berubah. Pada Q2, keluaran lengkap
memuat metrik yang sama dengan baseline, tetapi kelengkapan jawaban
belum konsisten pada seluruh pengulangan.

### Kesimpulan Experiment Temperature

Temperature memengaruhi generation, yaitu cara LLM menyusun jawaban.
Parameter ini tidak mengubah retrieval, pembentukan chunk, embedding,
atau vector database. Temperature 0.2 menjadi pilihan paling dapat
dipertanggungjawabkan karena menghasilkan Q1 dan Q2 lengkap pada
baseline. Hasil Q2 pada temperature 0.0 dan 0.8 belum konsisten.

------------------------------------------------------------------------

## 5. Experiment 2: TOP_K

### TOP_K = 1

**Actual Output Q1:** Aplikasi menampilkan pesan bahwa jawaban tidak
ditemukan dalam dokumen. Sumber: halaman 2. Similarity score: 0,8824.

**Actual Output Q2:** Jawaban hanya menyampaikan kesimpulan umum bahwa
Random Forest Regression lebih baik karena RMSE lebih rendah. Seluruh
metrik tidak disajikan. Sumber: halaman 7. Similarity score: 0,8225.

**Observation:** Satu konteks belum cukup untuk menjawab kedua
pertanyaan secara lengkap, meskipun similarity score dokumen yang
terambil relatif tinggi.

### TOP_K = 4

**Actual Output Q1:** Aplikasi menemukan lima variabel independen, yaitu
suhu udara, kelembapan udara, kecepatan angin, curah hujan, dan jumlah
titik panas. Variabel dependen adalah konsentrasi PM2,5. Sumber utama:
halaman 2. Similarity score khusus TOP_K = 4 tidak tercatat.

**Actual Output Q2:**

| Metrik       | Regresi Linier Berganda | Random Forest Regression |
|:-------------|------------------------:|-------------------------:|
| R²           |                   0,585 |                    0,588 |
| RMSE (µg/m³) |                   6,904 |                    6,756 |
| MAE (µg/m³)  |                   5,672 |                    5,490 |
| MAPE (%)     |                    26,1 |                     26,0 |
| Akurasi (%)  |                    73,9 |                     74,0 |

Sumber: halaman 7, 2, 6, dan 5. Halaman 6 menjadi sumber utama Tabel 2.
Similarity score khusus TOP_K = 4 tidak tercatat.

**Observation:** TOP_K = 4 menghasilkan jawaban Q1 dan Q2 secara
lengkap. Konteks yang diambil sudah mencakup informasi variabel
penelitian dan tabel perbandingan algoritma.

### TOP_K = 8

**Actual Output Q1:** Aplikasi menemukan lima variabel independen dan
variabel dependen konsentrasi PM2,5.

| Halaman | Similarity score |
|--------:|-----------------:|
|       2 |           0,8824 |
|       1 |           0,8308 |
|       3 |           0,7539 |
|       5 |           0,6801 |

**Actual Output Q2:** Hasil metrik sama dengan TOP_K = 4.

| Metrik       | Regresi Linier Berganda | Random Forest Regression |
|:-------------|------------------------:|-------------------------:|
| R²           |                   0,585 |                    0,588 |
| RMSE (µg/m³) |                   6,904 |                    6,756 |
| MAE (µg/m³)  |                   5,672 |                    5,490 |
| MAPE (%)     |                    26,1 |                     26,0 |
| Akurasi (%)  |                    73,9 |                     74,0 |

| Halaman | Similarity score |
|--------:|-----------------:|
|       7 |           0,8229 |
|       2 |           0,8168 |
|       6 |           0,8070 |
|       5 |           0,7865 |

**Observation:** TOP_K = 8 menghasilkan jawaban lengkap, tetapi tidak
menambahkan informasi faktual yang berarti dibandingkan TOP_K = 4.

### Kesimpulan Experiment TOP_K

TOP_K = 1 belum menyediakan konteks yang cukup. TOP_K = 4 dan TOP_K = 8
sama-sama menghasilkan jawaban lengkap. TOP_K = 8 mengambil lebih banyak
konteks, tetapi tidak memberi tambahan informasi faktual yang berarti.
TOP_K = 4 lebih efisien dan memadai untuk dokumen ini.

------------------------------------------------------------------------

## 6. Experiment 3: Chunk Size

Pengujian ini menggunakan CHUNK_OVERLAP = 20. Setiap perubahan
CHUNK_SIZE diikuti penghapusan database Qdrant dan ingestion ulang PDF.

### CHUNK_SIZE = 50

**Chunks Stored:** 221.

**Actual Output Q1:** Aplikasi berhasil menemukan lima variabel
independen dan variabel dependen konsentrasi PM2,5. Sumber: halaman 2,
7, dan 1.

**Actual Output Q2:** Jawaban hanya menampilkan sebagian metrik, yaitu
R² sebesar 0,585 dan 0,588 serta RMSE sebesar 6,904 dan 6,756. MAE,
MAPE, dan akurasi tidak muncul secara lengkap. Sumber: halaman 5, 6, dan
7. Similarity score tidak tercatat.

**Observation:** CHUNK_SIZE = 50 menghasilkan chunk paling banyak, yaitu
221. Namun, jumlah chunk yang lebih banyak tidak otomatis meningkatkan
kelengkapan jawaban. Q2 belum menampilkan seluruh metrik.

### CHUNK_SIZE = 100

**Chunks Stored:** 85.

**Actual Output Q1:** Aplikasi berhasil mengidentifikasi lima variabel
independen dan satu variabel dependen. Sumber utama: halaman 2.

**Actual Output Q2:** Aplikasi berhasil menampilkan seluruh nilai R²,
RMSE, MAE, MAPE, dan akurasi seperti tabel baseline. Sumber baseline
yang tercatat: halaman 7, 2, 6, dan 5. Similarity score khusus percobaan
ini tidak tercatat secara terpisah.

**Observation:** CHUNK_SIZE = 100 menghasilkan 85 chunk dan dapat
menjawab Q1 serta Q2 secara lengkap. Konfigurasi ini menjadi baseline
yang berhasil.

### CHUNK_SIZE = 120

**Chunks Stored:** 71.

**Actual Output Q1:** Aplikasi berhasil mengidentifikasi lima variabel
independen dan variabel dependen konsentrasi PM2,5. Sumber: halaman 1
dan 2.

**Actual Output Q2:** Aplikasi berhasil menampilkan seluruh metrik kedua
algoritma. Sumber utama: halaman 6, dengan referensi tambahan halaman 2,
5, dan 7. Similarity score tidak tercatat.

**Observation:** CHUNK_SIZE = 120 menghasilkan 71 chunk, lebih sedikit
daripada CHUNK_SIZE = 100. Q1 dan Q2 tetap berhasil dijawab secara
lengkap.

### Kesimpulan Experiment Chunk Size

CHUNK_SIZE = 50 menghasilkan 221 chunk, CHUNK_SIZE = 100 menghasilkan 85
chunk, dan CHUNK_SIZE = 120 menghasilkan 71 chunk. CHUNK_SIZE = 50 belum
memberikan jawaban Q2 lengkap, sedangkan CHUNK_SIZE = 100 dan 120
berhasil menjawab kedua pertanyaan. Jumlah chunk paling banyak tidak
selalu menghasilkan retrieval terbaik.

------------------------------------------------------------------------

## 7. Experiment 4: Chunk Overlap

Pengujian ini menggunakan CHUNK_SIZE = 100. Setiap perubahan
CHUNK_OVERLAP diikuti penghapusan database Qdrant dan ingestion ulang
PDF.

### CHUNK_OVERLAP = 0

**Chunks Stored:** 84.

**Actual Output Q1:** Aplikasi belum berhasil menemukan jawaban secara
memadai. Referensi yang tercatat mengarah ke halaman 2 dan 5.

**Actual Output Q2:** Aplikasi berhasil menampilkan seluruh metrik
perbandingan kedua algoritma. Sumber utama: halaman 6. Similarity score
tidak tercatat.

**Observation:** Tanpa overlap, Q1 gagal dijawab secara memadai,
meskipun Q2 berhasil.

### CHUNK_OVERLAP = 20

**Chunks Stored:** 85.

**Actual Output Q1:** Aplikasi berhasil menemukan lima variabel
independen dan variabel dependen konsentrasi PM2,5. Sumber utama:
halaman 2.

**Actual Output Q2:** Aplikasi berhasil menampilkan seluruh metrik
perbandingan kedua algoritma. Sumber utama: halaman 6. Referensi
baseline lain yang tercatat adalah halaman 7, 2, dan 5. Similarity score
khusus percobaan ini tidak tercatat.

**Observation:** Overlap 20 mempertahankan sebagian teks di antara chunk
berurutan. Pada percobaan ini, Q1 dan Q2 berhasil dijawab lengkap.

### CHUNK_OVERLAP = 40

**Chunks Stored:** 101.

**Actual Output Q1:** Aplikasi berhasil menemukan variabel independen
dan dependen. Sumber: halaman 1 dan 2.

**Actual Output Q2:** Aplikasi berhasil menampilkan seluruh metrik
perbandingan kedua algoritma. Sumber utama: halaman 6. Similarity score
tidak tercatat.

**Observation:** Overlap 40 menghasilkan 101 chunk, lebih banyak
dibandingkan overlap 0 dan 20. Q1 dan Q2 berhasil dijawab, tetapi jumlah
chunk meningkat.

### Kesimpulan Experiment Chunk Overlap

CHUNK_OVERLAP = 0 menghasilkan 84 chunk, CHUNK_OVERLAP = 20 menghasilkan
85 chunk, dan CHUNK_OVERLAP = 40 menghasilkan 101 chunk. Q1 gagal pada
overlap 0, sedangkan overlap 20 dan 40 berhasil. Q2 berhasil pada ketiga
variasi. Untuk dokumen ini, overlap 20 memberikan keseimbangan yang baik
antara jumlah chunk dan kelengkapan jawaban.

------------------------------------------------------------------------

## 8. Final Configuration

| Parameter       |                                                   Nilai akhir |
|:----------------|--------------------------------------------------------------:|
| Temperature     |                                                           0.2 |
| TOP_K           |                                                             4 |
| CHUNK_SIZE      |                                                           100 |
| CHUNK_OVERLAP   |                                                            20 |
| Jumlah chunks   |                                                            85 |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| LLM model       |                                         `openai/gpt-oss-120b` |

Konfigurasi ini dipilih karena merupakan baseline yang telah
menghasilkan jawaban Q1 dan Q2 secara lengkap. TOP_K = 4 menyediakan
konteks yang cukup tanpa tambahan konteks yang tidak memberi informasi
faktual baru seperti pada TOP_K = 8. CHUNK_SIZE = 100 dan CHUNK_OVERLAP
= 20 menghasilkan jawaban lengkap dengan 85 chunk.

CHUNK_SIZE = 120 menjadi alternatif karena menghasilkan 71 chunk dan
tetap menjawab Q1 serta Q2 secara lengkap. Namun, catatan similarity
score belum cukup untuk menyatakan bahwa konfigurasi tersebut lebih
unggul daripada baseline.

## 9. Final Questions

**1. Parameter mana yang paling terasa memengaruhi hasil aplikasi?**

CHUNK_SIZE, CHUNK_OVERLAP, dan TOP_K paling terlihat memengaruhi
kelengkapan konteks yang diterima LLM. TOP_K = 1 belum menghasilkan
jawaban lengkap. CHUNK_SIZE = 50 menghasilkan banyak chunk, tetapi Q2
belum menampilkan seluruh metrik. CHUNK_OVERLAP = 0 juga belum berhasil
menjawab Q1. Kualitas retrieval bergantung pada jumlah dan susunan
konteks yang diambil.

**2. Apakah Temperature memengaruhi Retrieval atau Generation?**

Temperature memengaruhi generation, yaitu proses LLM menyusun jawaban
berdasarkan konteks yang diterima. Temperature tidak mengubah proses
retrieval, pembentukan embedding, atau penyimpanan vector. Isi faktual
Q1 tetap sama pada beberapa nilai temperature. Perbedaan lebih terlihat
pada penyajian dan kelengkapan Q2.

**3. Apakah TOP_K yang lebih besar selalu menghasilkan jawaban yang
lebih baik?**

Tidak. TOP_K = 1 belum memberikan konteks yang cukup. TOP_K = 4 dan 8
sama-sama menghasilkan jawaban lengkap, tetapi TOP_K = 8 tidak
menambahkan informasi faktual yang berarti dibandingkan TOP_K = 4.

**4. Apa yang terjadi ketika Chunk Size dibuat lebih kecil?**

Jumlah chunk meningkat karena dokumen terbagi menjadi bagian lebih
pendek. CHUNK_SIZE = 50 menghasilkan 221 chunk, sedangkan CHUNK_SIZE =
100 menghasilkan 85 chunk. Pada pengujian ini, CHUNK_SIZE = 50 belum
menghasilkan jawaban Q2 lengkap. Chunk terlalu kecil dapat memisahkan
informasi yang saling berkaitan.

**5. Apa fungsi Chunk Overlap?**

Chunk Overlap mempertahankan sebagian teks dari chunk sebelumnya pada
chunk berikutnya. Cara ini membantu menjaga hubungan antarinformasi
ketika batas pemotongan dokumen berada di tengah pembahasan. Pada
eksperimen ini, overlap 20 dan 40 berhasil menjawab Q1, sedangkan
overlap 0 belum berhasil.

**6. Mengapa perubahan Chunk Size dan Chunk Overlap membutuhkan
ingestion ulang?**

Kedua parameter menentukan cara aplikasi membagi dokumen menjadi chunk.
Perubahan nilainya mengubah isi dan batas setiap chunk. Aplikasi perlu
membentuk ulang chunk, membuat embedding baru, dan menyimpan hasilnya
kembali ke Qdrant. Karena itu, database lama perlu dihapus sebelum
ingestion ulang agar hasil konfigurasi sebelumnya tidak bercampur.

**7. Mengapa perubahan Temperature tidak membutuhkan ingestion ulang?**

Temperature bekerja pada tahap generation ketika LLM menyusun jawaban.
Parameter ini tidak mengubah teks hasil ekstraksi, chunk, embedding,
atau vector yang tersimpan di Qdrant. Karena itu, perubahan temperature
tidak membutuhkan ingestion ulang.

**8. Apakah ada satu konfigurasi RAG yang selalu paling baik untuk semua
jenis dokumen?**

Tidak. Kualitas konfigurasi RAG bergantung pada struktur dokumen,
panjang informasi, jenis pertanyaan, model embedding, model LLM, dan
kebutuhan pengguna. Konfigurasi yang bekerja baik pada artikel PM2,5
belum tentu memberikan hasil sama pada dokumen lain. Pengguna perlu
menguji parameter berdasarkan karakteristik dokumen dan pertanyaan.

## 10. Personal Reflection

**Bagian eksperimen mana yang hasilnya paling berbeda dari dugaan?**

Pengujian CHUNK_SIZE = 50 memberikan hasil yang paling berbeda. Jumlah
chunk meningkat menjadi 221, tetapi jawaban Q2 belum menampilkan seluruh
metrik. Jumlah chunk lebih banyak tidak otomatis meningkatkan
kelengkapan jawaban.

**Parameter mana yang awalnya dianggap tidak terlalu penting tetapi
ternyata berpengaruh?**

CHUNK_OVERLAP. Pada overlap 0, aplikasi belum berhasil menjawab Q1
secara memadai. Ketika overlap menjadi 20 dan 40, Q1 berhasil dijawab.
Bagian teks yang berulang di antara chunk dapat membantu mempertahankan
konteks.

**Bagaimana hubungan Retrieval dan Generation pada RAG?**

Retrieval mencari dan mengambil potongan dokumen yang relevan dengan
pertanyaan. Generation menggunakan potongan tersebut untuk menyusun
jawaban. Keduanya saling bergantung. Jika retrieval tidak mengambil
informasi yang dibutuhkan, generation dapat memberikan jawaban tidak
lengkap meskipun model LLM mampu menyusun kalimat dengan baik.

**Jika aplikasi dikembangkan lebih lanjut, bagian apa yang ingin
diuji?**

Pengembangan berikutnya dapat menguji reranking untuk mengurutkan ulang
hasil retrieval, hybrid search untuk menggabungkan pencarian semantik
dan kata kunci, serta evaluasi retrieval menggunakan kumpulan pertanyaan
dan jawaban acuan. Pengujian tersebut dapat membantu menilai relevansi
konteks dan ketepatan jawaban secara lebih terukur.

## 11. AI Usage

**AI tools yang digunakan:** ChatGPT.

**AI digunakan untuk membantu bagian apa?**

ChatGPT membantu memahami instruksi eksperimen, menjelaskan konsep
Retrieval-Augmented Generation, merapikan dokumentasi hasil,
membandingkan pengaruh parameter, dan menyusun refleksi berdasarkan
hasil yang diperoleh.

**Bagian eksperimen mana yang dilakukan dan dievaluasi sendiri?**

Pengguna menjalankan aplikasi Simple RAG pada komputer sendiri, mengubah
parameter pada file konfigurasi, melakukan ingestion ulang ketika
mengubah parameter chunking, mengunggah PDF, mengajukan Q1 dan Q2, serta
mengamati jawaban dan referensi sumber pada aplikasi. ChatGPT membantu
mengorganisasi dan menjelaskan hasil tersebut. Nilai eksperimen dalam
README mengikuti hasil yang tercatat dari aplikasi.

## 12. Catatan Data dan Batasan

1.  Similarity score hanya dicantumkan pada percobaan yang memiliki
    catatan angka. Nilai yang tidak terlihat pada hasil aplikasi tidak
    dibuat atau diperkirakan.
2.  Beberapa pengulangan temperature 0.8 menghasilkan keluaran Q2 yang
    berbeda kelengkapannya. Karena itu, kelengkapan Q2 pada temperature
    0.8 belum dapat dinyatakan konsisten.
3.  Pada beberapa variasi chunk, catatan sumber hanya memuat halaman
    utama atau rentang halaman. README tidak menambahkan similarity
    score yang tidak tersedia.
4.  Konfigurasi final menggunakan baseline yang telah menghasilkan
    jawaban lengkap untuk Q1 dan Q2. Pemilihan ini didasarkan pada bukti
    hasil eksperimen, bukan klaim bahwa konfigurasi tersebut paling baik
    untuk semua dokumen.

## 13. Checklist Sebelum Pengumpulan

- [ ] Aplikasi berhasil dijalankan.
- [ ] PDF berhasil di-ingest.
- [ ] Q1 dan Q2 berhasil diuji.
- [ ] Eksperimen Temperature selesai.
- [ ] Eksperimen TOP_K selesai.
- [ ] Eksperimen CHUNK_SIZE selesai.
- [ ] Eksperimen CHUNK_OVERLAP selesai.
- [ ] Actual output dan sumber dicatat.
- [ ] Similarity score dicatat jika tersedia.
- [ ] Jumlah `chunks_stored` dicatat.
- [ ] README.md sudah diperiksa.
- [ ] Final configuration dan refleksi sudah dicantumkan.
- [ ] File `.env` tidak ikut diunggah.
- [ ] Folder `venv/`, `qdrant_data/`, `data/uploads/`, dan
  `__pycache__/` tidak ikut diunggah.
- [ ] Repository GitHub `tugas-rag-experiment` sudah dibuat dan berisi
  source code serta README.
- [ ] Link repository siap dikumpulkan ke LMS.
