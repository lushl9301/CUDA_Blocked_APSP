#ifndef _APSP_H_
#define _APSP_H_

#define REFERENCE
void GenMatrix(int *mat, const size_t N);
bool CmpArray(const int *l, const int *r, const size_t eleNum);
void ST_APSP(int *mat, const size_t N);

void GenMatrix(int *mat, const size_t N)
{
	for (int i = 0; i < N * N; i++) {
		int temp = rand() % 32;
		if (temp == 0)
			temp = 10000000;
		mat[i] = temp - 1;
	}
	for (int i = 0; i < N; i++)
		mat[i * N + i] = 0;
}

bool CmpArray(const int *l, const int *r, const size_t eleNum)
{
	for (int i = 0; i < eleNum; i++)
		if (l[i] != r[i]) {
			printf("ERROR: l[%d] = %d, r[%d] = %d\n", i, l[i], i, r[i]);
			return false;
		}
	return true;
}


/*
        Sequential (Single Thread) APSP on CPU.
 */
void ST_APSP(int *mat, const size_t N)
{
	for (int k = 0; k < N; k++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				int i0 = i * N + j;
				int i1 = i * N + k;
				int i2 = k * N + j;
				int sum =  (mat[i1] + mat[i2]);
				if (sum < mat[i0])
					mat[i0] = sum;
			}
}

#endif
