// a131_kernel.cpp — a1_3_1 구조격자(ring x column) 이웃 표와 면방향 PCA 의 모멘트 누적을 C++ 로(2026-11-13).
// 파이썬 a131_vec.neighbor_table 과 a131_live.support_pca_fast 의 누적 루프와 '같은 연산 순서'로 계산한다.
//   - 음수 인덱스는 numpy 처럼 뒤에서부터(idx + N) 읽는다.
//   - np.maximum 처럼 NaN 을 전파한다.
//   - 컴파일: g++ -O3 -fPIC -shared -ffp-contract=off (곱셈-덧셈을 합치지 않아야 numpy 와 끝자리까지 같다)
#include <cstdint>
#include <cmath>
#include <cstring>

static inline int64_t wrap(int64_t i, int64_t n) { return i < 0 ? i + n : i; }

extern "C" {

// 1단계: reverse_ring / reverse_col 을 만들고 중심점(centers, r0, c0)을 채운다. 반환 = 중심점 수 M.
//   centers/r0/c0 는 길이 point_count 버퍼. rev_ring/rev_col 은 길이 point_count 작업 버퍼.
int64_t a131_centers(const int64_t* grid, int64_t ring_count, int64_t column_count,
                     const uint8_t* valid, int64_t point_count,
                     int64_t* rev_ring, int64_t* rev_col,
                     int64_t* centers, int64_t* r0, int64_t* c0) {
    for (int64_t i = 0; i < point_count; ++i) { rev_ring[i] = -1; rev_col[i] = -1; }
    for (int64_t r = 0; r < ring_count; ++r) {
        const int64_t* row = grid + r * column_count;
        for (int64_t c = 0; c < column_count; ++c) {            // numpy 의 C 순서 대입(같은 인덱스가 여러 번이면 마지막 값)
            int64_t p = wrap(row[c], point_count);
            rev_ring[p] = r; rev_col[p] = c;
        }
    }
    int64_t m = 0;
    for (int64_t i = 0; i < point_count; ++i) {
        if (valid[i] && rev_ring[i] >= 0 && rev_col[i] >= 0) { centers[m] = i; r0[m] = rev_ring[i]; c0[m] = rev_col[i]; ++m; }
    }
    return m;
}

// 2단계: 이웃 표. nidx/avail/acc 는 (K, M) C 순서. offsets 는 (K, 2) = (dr, dc).
void a131_table(const int64_t* grid, int64_t ring_count, int64_t column_count,
                const uint8_t* valid, const double* rng, int64_t point_count,
                const int64_t* centers, const int64_t* r0, const int64_t* c0, int64_t m,
                const int64_t* offsets, int64_t k_count, double jump_base, double jump_ratio,
                double* crange, double* jump, int64_t* nidx, uint8_t* avail, uint8_t* acc) {
    for (int64_t j = 0; j < m; ++j) {
        crange[j] = rng[centers[j]];
        jump[j] = jump_base + jump_ratio * crange[j];
    }
    for (int64_t k = 0; k < k_count; ++k) {
        const int64_t dr = offsets[2 * k], dc = offsets[2 * k + 1];
        int64_t* nk = nidx + k * m; uint8_t* ak = avail + k * m; uint8_t* ck = acc + k * m;
        for (int64_t j = 0; j < m; ++j) {
            int64_t rr = r0[j] + dr;
            bool inside = (rr >= 0) && (rr < ring_count);
            int64_t cc = (c0[j] + dc) % column_count; if (cc < 0) cc += column_count;   // 파이썬 % (음수 아님)
            int64_t rc = rr < 0 ? 0 : (rr > ring_count - 1 ? ring_count - 1 : rr);
            int64_t n = grid[rc * column_count + cc];
            int64_t nw = wrap(n, point_count);
            bool a = inside && valid[nw];
            nk[j] = n; ak[j] = a ? 1 : 0;
            // 원본: abs(range[n] - center_range) > jump 이면 거부 (NaN 이면 비교가 거짓 → 수용)
            ck[j] = (a && !(std::fabs(rng[nw] - crange[j]) > jump[j])) ? 1 : 0;
        }
    }
}

// 3단계: 지지점 [중심, 수용 이웃] 의 중심 기준 상대좌표 1·2차 모멘트와 최대 반경^2.
//   cols(길이 q) = fit_mask 인 중심의 열 번호. 출력 10개는 길이 q, 0 으로 시작해 오프셋 순서대로 누적.
void a131_moments(const double* points, int64_t point_count,
                  const int64_t* centers, const int64_t* nidx, const uint8_t* acc, int64_t k_count, int64_t m,
                  const int64_t* cols, int64_t q,
                  double* sx, double* sy, double* sz, double* sxx, double* syy, double* szz,
                  double* sxy, double* sxz, double* syz, double* r2max) {
    for (int64_t i = 0; i < q; ++i) { sx[i] = sy[i] = sz[i] = sxx[i] = syy[i] = szz[i] = sxy[i] = sxz[i] = syz[i] = r2max[i] = 0.0; }
    for (int64_t k = 0; k < k_count; ++k) {
        const int64_t* nk = nidx + k * m; const uint8_t* ck = acc + k * m;
        bool any = false;
        for (int64_t i = 0; i < q; ++i) if (ck[cols[i]]) { any = true; break; }
        if (!any) continue;                                      // 파이썬: if not a.any(): continue
        for (int64_t i = 0; i < q; ++i) {
            const int64_t j = cols[i];
            double dx = 0.0, dy = 0.0, dz = 0.0;
            if (ck[j]) {
                const double* pc = points + 3 * centers[j];
                const double* pn = points + 3 * wrap(nk[j], point_count);
                dx = pn[0] - pc[0]; dy = pn[1] - pc[1]; dz = pn[2] - pc[2];
            }
            sx[i] += dx; sy[i] += dy; sz[i] += dz;
            sxx[i] += dx * dx; syy[i] += dy * dy; szz[i] += dz * dz;
            sxy[i] += dx * dy; sxz[i] += dx * dz; syz[i] += dy * dz;
            const double r2 = dx * dx + dy * dy + dz * dz;
            // np.maximum(r2max, r2): 둘 중 하나가 NaN 이면 NaN
            if (!std::isnan(r2max[i])) { if (std::isnan(r2) || r2 > r2max[i]) r2max[i] = r2; }
        }
    }
}

}  // extern "C"
