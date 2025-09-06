# Business Problem Analysis - Dataset Iris

## 1. Latar Belakang Masalah

Dalam industri hortikultura dan penelitian botani, identifikasi spesies bunga Iris secara manual memerlukan keahlian khusus dan waktu yang signifikan. Proses identifikasi tradisional bergantung pada pengalaman ahli botani yang terbatas dan subjektif, yang dapat mengakibatkan:

- **Inkonsistensi identifikasi** antar berbagai ahli
- **Waktu identifikasi yang lama** (5-10 menit per sampel)
- **Biaya operasional tinggi** untuk pelatihan dan konsultasi ahli
- **Kesalahan klasifikasi** yang berdampak pada penelitian dan bisnis

## 2. Pernyataan Masalah Bisnis

**"Bagaimana mengembangkan sistem klasifikasi otomatis yang akurat dan efisien untuk mengidentifikasi spesies bunga Iris berdasarkan pengukuran morfologi yang mudah dilakukan di lapangan?"**

### Sub-masalah:
- Fitur pengukuran mana yang paling efektif untuk membedakan spesies?
- Seberapa akurat sistem otomatis dibanding identifikasi manual?
- Bagaimana mengimplementasikan sistem ini dalam aplikasi praktis?

## 3. Tujuan Bisnis

### Primary Objectives:
- Mengembangkan model prediktif dengan akurasi ≥ 95%
- Mengurangi waktu identifikasi menjadi < 30 detik
- Mengotomatisasi proses klasifikasi Iris

### Secondary Objectives:
- Mengidentifikasi fitur pembeda yang paling signifikan
- Menyediakan insights untuk pengembangan aplikasi mobile
- Menciptakan standard operating procedure untuk pengukuran

## 4. Stakeholder Analysis

### Primary Stakeholders:
- **Peneliti Botani**: Membutuhkan tools identifikasi cepat dan akurat
- **Nursery Business**: Automasi untuk inventory dan quality control
- **Educational Institutions**: Material pembelajaran dan praktikum

### Secondary Stakeholders:
- **App Developers**: Fitur plant identification dalam aplikasi
- **Garden Centers**: Tools untuk customer service
- **Conservation Organizations**: Monitoring biodiversitas

## 5. Value Proposition

### Manfaat Ekonomi:
- **Pengurangan biaya operasional**: 60-80% dari biaya identifikasi manual
- **Peningkatan throughput**: 10x lebih cepat dari metode tradisional
- **Konsistensi hasil**: Eliminasi variabilitas antar pengamat

### Manfaat Teknis:
- **Standardisasi proses**: Protocol pengukuran yang seragam
- **Scalability**: Dapat diimplementasikan di berbagai platform
- **Accessibility**: Tidak memerlukan keahlian khusus

## 6. Success Metrics

### Quantitative Metrics:
- **Akurasi Model**: ≥ 95% accuracy rate
- **Processing Speed**: < 1 detik prediction time
- **Cost Reduction**: 70% pengurangan biaya identifikasi
- **User Adoption**: 80% acceptance rate dari target users

### Qualitative Metrics:
- **Ease of Use**: User satisfaction score ≥ 4.5/5
- **Reliability**: Consistent results across different conditions
- **Interpretability**: Clear explanation of classification decisions

## 7. Constraints & Assumptions

### Technical Constraints:
- Dataset terbatas pada 3 spesies Iris
- Pengukuran manual dengan tingkat error ±0.1 cm
- Model harus dapat dijalankan pada mobile device

### Business Constraints:
- Budget development terbatas
- Timeline implementasi 6 bulan
- Harus compatible dengan existing workflows

### Assumptions:
- Pengukuran morfologi cukup untuk klasifikasi akurat
- User akan mengikuti protocol pengukuran yang ditetapkan
- Market demand untuk automated plant identification tinggi

## 8. Risk Assessment

### High Risk:
- **Model overfitting** pada dataset kecil
- **Measurement errors** yang mempengaruhi akurasi
- **User adoption resistance** dari traditional botanists

### Medium Risk:
- **Technology compatibility** issues
- **Regulatory compliance** untuk commercial use
- **Competition** dari existing solutions

### Mitigation Strategies:
- Cross-validation dan external testing
- Comprehensive user training program
- Phased implementation approach

## 9. Implementation Roadmap

### Phase 1 (Bulan 1-2): Data Analysis & Model Development
- Analisis komprehensif dataset Iris
- Feature engineering dan selection
- Model training dan validation

### Phase 2 (Bulan 3-4): Prototype Development
- Development of web-based prototype
- Database integration (MySQL/PostgreSQL)
- Power BI dashboard creation

### Phase 3 (Bulan 5-6): Testing & Deployment
- User acceptance testing
- Performance optimization
- Production deployment

## 10. Expected Outcomes

### Short-term (3-6 bulan):
- Working prototype dengan akurasi > 90%
- Validated measurement protocol
- Initial user feedback dan improvements

### Medium-term (6-12 bulan):
- Commercial-ready application
- Integration dengan existing systems
- Expanded species coverage

### Long-term (1-2 tahun):
- Market leadership dalam plant identification
- Revenue generation dari licensing
- Research collaboration opportunities

---

**Document Information:**
- **Author**: AchmadLutfi196
- **Date**: 2025-01-10
- **Version**: 1.0
- **Status**: Draft untuk Review