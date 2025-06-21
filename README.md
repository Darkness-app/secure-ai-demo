# secure-ai-demo

## 📋 Proje Açıklaması
Güvenli AI demo projesi

## 🔐 Güvenlik
Bu proje güvenli API kullanımı için tasarlanmıştır:
- API token'ları environment variable'larda saklanır
- Hassas bilgiler kod içinde bulunmaz
- Production-ready konfigürasyon

## 🛠️ Kurulum

### 1. Repository'yi klonlayın
```bash
git clone https://github.com/username/secure-ai-demo.git
cd secure-ai-demo
```

### 2. Virtual environment oluşturun
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 3. Paketleri yükleyin
```bash
pip install -r requirements.txt
```

### 4. Environment variables ayarlayın
```bash
cp .env.example .env
# .env dosyasını düzenleyin ve API token'larınızı ekleyin
```

### 5. Uygulamayı çalıştırın
```bash
python main.py
```

## 🚀 Kullanım

### API Endpoints

#### POST /api/process
Metin işleme endpoint'i
```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Merhaba AI!"}'
```

#### GET /api/health
Sistem durumu kontrolü
```bash
curl http://localhost:5000/api/health
```

## 🧪 Test

```bash
python test.py
```

## 📊 Özellikler
- ✅ Güvenli API token yönetimi
- ✅ AI model entegrasyonu
- ✅ RESTful API
- ✅ Health check endpoint
- ✅ Production-ready konfigürasyon

## 📝 Lisans
MIT License

---
*🤖 Bu proje AI Master System tarafından güvenli şekilde oluşturulmuştur.*
