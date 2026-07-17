# Portföy Takip

Kişisel hisse senedi / ETF yatırımlarını tek yerden takip etmek için basit bir
web uygulaması. Flask + SQLite ile yazıldı, kurulumu ve çalıştırması kolay.

## Nasıl çalıştırılır

```bash
cd portfolio-tracker
pip install -r requirements.txt
python app.py
```

Sonra tarayıcıdan `http://127.0.0.1:5000` adresine git.

## Veriler nasıl saklanıyor?

Uygulama ilk çalıştığında yanına `portfolio.db` adında bir dosya oluşturuyor.
Bu bir **SQLite veritabanı** — yani tek bir dosyanın içine gömülü, sunucu
kurmaya gerek olmayan bir veritabanı. Tüm işlemlerin (alım/satım kayıtların)
bu dosyada tutuluyor; tarayıcıyı kapatsan da, bilgisayarı yeniden başlatsan da
veriler kaybolmuyor. Yedeklemek istersen tek yapman gereken bu dosyayı
kopyalamak. (`.gitignore` içinde bu dosya hariç tutuldu, yani kişisel
verilerin git'e / GitHub'a gitmiyor.)

## Proje yapısı

```
portfolio-tracker/
├── app.py          # Flask uygulaması: URL'ler (route'lar) burada tanımlı
├── db.py           # Veritabanı bağlantısı ve tablo şeması (yapısı)
├── fifo.py         # Alım/satım eşleştirme mantığı (FIFO)
├── templates/       # Sayfaların HTML iskeletleri (Jinja2 şablonları)
├── static/          # CSS (görünüm) dosyaları
└── portfolio.db     # (otomatik oluşur) senin verilerinin tutulduğu dosya
```

## Adım 1 durumu: Veri modeli + İşlem Kaydı

Şu an eklenen:
- `transactions` tablosu: her alım/satım için tarih, sembol, tip, adet,
  fiyat, toplam, kategori/tema etiketi ve serbest metin not.
- `/transactions` sayfası: yeni işlem ekleme formu + tüm işlemlerin listesi
  (silme dahil).
- `fifo.py`: bir sembole ait işlemleri FIFO mantığıyla eşleştiren fonksiyon
  (`match_fifo`). Bu, henüz ekranda kullanılmıyor — 2. ve 3. adımlarda
  (portföy özeti ve pozisyon detayı) devreye girecek.

## Adım 2 durumu: Portföy Özeti

Şu an eklenen:
- `portfolio.py`: tüm sembolleri FIFO ile hesaplayıp pozisyon listesi ve
  toplamları çıkaran modül (gerçekleşmemiş/gerçekleşmiş K/Z, maliyet
  bazı, piyasa değeri).
- `/` (Özet) sayfası: toplam portföy değeri, gerçekleşmemiş K/Z,
  gerçekleşmiş K/Z (satılmış pozisyonlardan), nakit bakiyesi, net değer
  ve **girişten itibaren (all-time) performans** ($ ve %).
- Nakit hareketi ekleme formu (yatırım / çekim) — `cash_flows` tablosuna
  yazıyor. Nakit bakiyesi = yatırılan sermaye − çekilen − alımlar +
  satımlar.
- Pozisyonlar tablosu: her sembol için adet, ortalama maliyet, güncel
  fiyat, piyasa değeri, K/Z.

**Önemli:** Güncel fiyat API'si henüz bağlı değil, bu yüzden "güncel
fiyat" olarak o sembole ait **son işlem fiyatı** kullanılıyor (tabloda
"(son işlem fiyatı)" notuyla belirtiliyor). Aynı sebeple günlük /
haftalık / aylık / YTD performans karşılaştırmaları da henüz yok — bunlar
geçmiş piyasa fiyatı gerektiriyor ve 4. adımda (yfinance entegrasyonu)
eklenecek. Girişten itibaren (all-time) performans ise geçmiş fiyata
ihtiyaç duymadığı için şimdiden doğru hesaplanıyor.

Henüz **yok**: pozisyon detay sayfası (açık lotlar, sembole ait işlem
geçmişi), canlı fiyat, zaman dilimi performansı, etiket/tarih filtreleme.
