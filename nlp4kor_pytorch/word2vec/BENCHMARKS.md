# Training Time
- GTX1080Ti (GPU 11GB memory)
- batch=1000, embed=300, epoch=10, neg_sample=20, vocab_size=100,000
    - GPU: 3 hours
    - CPU: 3 days+

- batch=1000, embed=300, epoch=20, neg_sample=20, vocab_size=100,000
    - GPU: 6 hours

- batch=500, embed=300, epoch=20, neg_sample=100, vocab_size=100,000 (best)
    - GPU: 16 hours (1.7 GB)

- batch=1000, embed=300, epoch=10, neg_sample=20, vocab_size=1,000,000
    - GPU: out of memory
    - CPU: 540 hours (22days+)
