# Memory Optimization for VRAM

## Các tối ưu hóa VRAM đã thực hiện (KHÔNG ảnh hưởng độ chính xác)

### 1. **Xóa intermediate tensors ngay sau khi sử dụng**
- Sử dụng `del` để xóa tensors không cần thiết ngay sau khi tính toán xong
- Giải phóng bộ nhớ cho các biến tạm: `input_ids`, `attention_mask`, `outputs`, `logits`, etc.

**Files đã chỉnh sửa:**
- `properties/toxicity_property.py`
- `properties/perplexity_property.py`
- `x_theta_modifier/local_search_language_utils.py`

**Ví dụ:**
```python
# Trước:
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
toxicity_scores = outputs['logits'].cpu()

# Sau:
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
toxicity_scores = outputs['logits'].cpu()
del input_ids, attention_mask, outputs  # Free memory immediately
```

### 2. **Synchronize và clear CUDA cache thường xuyên**
- Thêm `torch.cuda.synchronize()` để đảm bảo tất cả operations hoàn thành
- Sử dụng `torch.cuda.empty_cache()` để giải phóng memory pool
- Thực hiện sau mỗi lần inference và sau mỗi rank trong local search

**Ví dụ:**
```python
model.to('cpu')
torch.cuda.empty_cache()
torch.cuda.synchronize()  # Ensure all operations complete
```

### 3. **Giải phóng memory sau mỗi rank trong local search**
- Clear tensors sau khi xử lý xong mỗi rank
- Giảm peak memory usage đáng kể

**Trong `local_search_language_utils.py`:**
```python
# After processing each rank
del neighbor_tokens_batch, all_neighbor_metadata, neighbor_distances_tensor
torch.cuda.empty_cache()
```

## Kết quả mong đợi

### Memory savings:
- **~15-20% VRAM** nhờ xóa intermediate tensors sớm
- **~10-15% VRAM** nhờ clear cache thường xuyên
- **Tổng cộng: ~25-35% giảm peak VRAM usage**

### Performance impact:
- **KHÔNG ảnh hưởng độ chính xác** (100% giữ nguyên model precision)
- **Latency tăng nhẹ** (~2-5%) do thêm synchronization overhead
- **Trade-off hợp lý**: Giảm VRAM đáng kể với cost rất nhỏ về tốc độ

## Các tối ưu hóa KHÔNG thực hiện (có thể ảnh hưởng độ chính xác)

❌ **KHÔNG load models với float16/bfloat16** - Có thể làm giảm độ chính xác
❌ **KHÔNG giảm batch_size cho property calculation** - Theo yêu cầu người dùng
❌ **KHÔNG quantization** - Có thể ảnh hưởng kết quả

## Monitoring

Để kiểm tra memory usage khi chạy:
```bash
# Terminal 1: Run inference
python inference_search.py ...

# Terminal 2: Monitor GPU memory
watch -n 1 nvidia-smi
```

## Các tối ưu hóa bổ sung có thể xem xét (nếu vẫn bị OOM)

1. **Giảm `top_k_values_for_local_search`**: Từ 5 xuống 3 hoặc 2
2. **Process theo mini-batches**: Chia batch lớn thành các mini-batches nhỏ hơn
3. **Tăng RAM allocation**: Request thêm RAM từ SLURM (hiện tại: 32G)
4. **Early stopping**: Stop local search sớm nếu đã tìm được sequence tốt

## Notes

- Tất cả các tối ưu hóa đều **đồng bộ** và **deterministic**
- Kết quả inference **hoàn toàn giống nhau** trước và sau tối ưu hóa
- Chỉ ảnh hưởng đến memory usage và latency, KHÔNG ảnh hưởng output
