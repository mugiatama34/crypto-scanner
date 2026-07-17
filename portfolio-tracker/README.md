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

## Adım 3 durumu: Pozisyon Detayı

Şu an eklenen:
- `/positions/<sembol>` sayfası (özet ekranındaki sembol adına
  tıklayınca açılıyor): net adet, ortalama maliyet, güncel fiyat,
  piyasa değeri, gerçekleşmemiş/gerçekleşmiş K/Z.
- **Açık Lotlar (FIFO)** tablosu: hangi tarihte, hangi fiyattan alınan
  lotların hâlâ elde olduğu.
- **Kapanan Eşleşmeler** tablosu: her satışın hangi alım lotunu/lotlarını
  kapattığı ve o eşleşmeden doğan gerçekleşen K/Z.
- **İşlem Geçmişi** tablosu: o sembole ait tüm alım/satımlar ve alım
  anında yazdığın notlar (tez/katalizör), en yeniden eskiye.
- Olmayan bir sembol için `/positions/...` adresine gidilirse 404
  dönüyor.

## Adım 4 durumu: Fiyat API Entegrasyonu

**Neden yfinance?** Üç seçeneği karşılaştırdım:

| Seçenek | Neden / neden değil |
|---|---|
| **yfinance (seçildi)** | API anahtarı gerekmiyor, hisse+ETF+kripto destekliyor, geçmiş fiyat verisini de (dönem karşılaştırmaları için şart) ücretsiz sağlıyor. |
| Alpha Vantage | Ücretsiz katman dakikada 5 / günde 25 istekle sınırlı — birkaç pozisyonu bile güncellemek yetersiz kalıyor. |
| Finnhub | Ücretsiz katmanda geçmiş (tarihsel) fiyat verisi yok — dönem karşılaştırmaları için kullanılamıyor. |

Şu an eklenen:
- `prices.py`: Yahoo Finance'ten güncel fiyat ve geçmiş kapanış
  fiyatlarını çeken modül. Ağ hatası / bulunamayan sembol durumunda
  sessizce boş/`None` döner — uygulama asla bu yüzden çökmez, sadece o
  sembol için "son işlem fiyatı" gösterilmeye devam eder.
- **Fiyat önbelleği** (`price_cache` tablosu): fiyatlar veritabanında
  saklanır, 4 saatten taze ise tekrar ağa gidilmez. Böylece her sayfa
  yenilemesi Yahoo Finance'e istek atmıyor.
- **"Fiyatları Yenile" butonu** (özet ekranında): önbelleği görmezden
  gelip anında güncel fiyat çeker; en son ne zaman güncellendiği de
  gösterilir.
- **Dönem karşılaştırmaları** (Günlük / Haftalık / Aylık / YTD): her
  dönemin başındaki net değeri (o tarihteki hisse adetleri × o tarihteki
  kapanış fiyatı + o tarihteki nakit) bugünkü net değerle kıyaslıyor.
  Dönem içinde yapılan yeni yatırım/çekimler bu kıyaslamadan
  ayıklanıyor — yoksa yeni para yatırmak sahte bir "kazanç" gibi
  görünürdü. Bir sembolün o tarihe ait fiyatı bulunamazsa (örn. ağ
  sorunu), o dönem için sayı yerine "Fiyat verisi yok" gösteriliyor —
  yanlış/uydurma bir rakam asla gösterilmiyor.
- Pozisyon detay sayfası da artık canlı fiyatı kullanıyor.

**Not (test ortamı hakkında):** Bu görevi yürüttüğüm sanal ortamın ağ
politikası Yahoo Finance'e erişimi engelliyor (kurumsal proxy 403
döndürüyor), bu yüzden gerçek bir Yahoo bağlantısını burada
gösteremedim. Tüm hesaplama mantığını (önbellekleme, dönem
karşılaştırmaları, FIFO ile geçmiş pozisyon hesabı) sahte/mock fiyat
verisiyle uçtan uca test ettim ve elle hesapladığım beklenen sonuçlarla
birebir eşleştiğini doğruladım. Kendi bilgisayarında normal internet
erişimiyle çalıştırdığında `yfinance` gerçek Yahoo Finance verisini
çekecektir — ek bir ayar gerekmez.

## Adım 5 durumu: Etiketleme ve Filtreleme (tamamlandı)

- Özet ekranındaki **Pozisyonlar** tablosunda etiket filtresi (dropdown,
  seçince otomatik uyguluyor).
- **İşlemler** ekranında etiket filtresi + tarih filtresi (Bu hafta / Bu
  ay / Bu yıl / Tümü — takvim bazlı: "bu hafta" pazartesiden bugüne,
  "bu ay" ayın 1'inden bugüne, "bu yıl" 1 Ocak'tan bugüne).
- Etiketler zaten işlem eklerken serbest metin olarak giriliyordu (adım
  1); bu adımda sadece bunları filtrelemek için arayüz eklendi, veri
  modelinde değişiklik yok.

---

## MVP tamamlandı

Başlangıçta planlanan 4 adım + etiketleme/filtreleme tamamlandı. Özetle
elindeki uygulama:
- İşlem kaydı (FIFO eşleştirmeli, notlu)
- Portföy özeti (toplam değer, gerçekleşmemiş/gerçekleşmiş K/Z, nakit,
  girişten itibaren ve günlük/haftalık/aylık/YTD performans)
- Pozisyon detayı (açık lotlar, kapanan eşleşmeler, işlem geçmişi/notlar)
- yfinance ile güncel fiyat (önbellekli + manuel yenile)
- Etiket ve tarih filtreleme

**Kendi bilgisayarında denemek için:**
```bash
cd portfolio-tracker
pip install -r requirements.txt
python app.py
```
sonra `http://127.0.0.1:5000` adresine git. İlk açılışta `portfolio.db`
otomatik oluşur; verilerin orada kalıcı olarak saklanır.
