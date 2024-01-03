#include "luca.h"

struct DBColumn {
public:
    int *data;
    int size;

private:
    DBColumn(int *ptr, int dataSize) {
      data = ptr;
      size = dataSize;
    }

public:
    DBColumn(int dataSize) {
      data = new int[dataSize];
      size = dataSize;
    }

    void fill(int value) {
      fill(0, size, value);
    }

    void fill(int start, int end, int value) {
      for (int i=start; i<end; i++)
        data[i] = value;
    }

    void randomize() {
      randomize(0, size, INT_MIN, INT_MAX);
    }

    void randomize(int min, int max) {
      randomize(0, size, min, max);
    }

    void randomize(int start, int end, int min, int max) {
        std::random_device rd;
        std::mt19937 gen(rd()); // Mersenne Twister generator

        // Define a distribution range
        std::uniform_int_distribution<> dis(min, max);

          for (int i=start; i<end; i++)
            data[i] = dis(gen);
        }

    DBColumn *view(int start, int end) {
        assert(start >= 0);
        assert(end > start);
        assert(end <= size);

        return new DBColumn(data + start, end-start);
    }
};

class DBEngine {
  protected:
    DBColumn *col;

  public:
    DBEngine(DBColumn *column): col(column) { }
    virtual int filterCount(int minIncluded, int maxExcluded) = 0;
    virtual double average() = 0;
    virtual int min() = 0;
    virtual int max() = 0;

    virtual const char* name() = 0;

    // Can execute some particular initialization, like allocating data in the GPU
    virtual void init() { }
    // Can copy memory into the GPU
    virtual void copy() { }

    virtual ~DBEngine() {}
};

class DBEngineCPU: public DBEngine {
  public:
    DBEngineCPU(DBColumn *column): DBEngine(column) {  }

    virtual int filterCount(int minIncluded, int maxExcluded) {
        int count = 0;

        for (int i=0; i<col->size; i++) {
          if (col->data[i] >= minIncluded && col->data[i] < maxExcluded)
            count++;
        }

        return count;
    }

    virtual double average() {
        long sum = 0;

        for (int i=0; i<col->size; i++) {
            sum+=col->data[i];
        }

        return sum / (double) col->size;
    }

    virtual int min() {
        int min = INT_MAX;

        for (int i=0; i<col->size; i++) {
          if (col->data[i] < min)
            min = col->data[i];
        }

        return min;
    }
    virtual int max() {
        int max = INT_MIN;

        for (int i=0; i<col->size; i++) {
          if (col->data[i] > max)
            max = col->data[i];
        }

        return max;
    }

    virtual const char* name() {
      return "CPU";
    }
};

class DBEngineCPU4Threads: public DBEngine {
  private:
    DBEngineCPU *cpu1;
    DBEngineCPU *cpu2;
    DBEngineCPU *cpu3;
    DBEngineCPU *cpu4;
    int threadSize;

  public:
    DBEngineCPU4Threads(DBColumn *column): DBEngine(column) {
        assert(column->size % 4 == 0);
        threadSize = column->size / 4;

        cpu1 = new DBEngineCPU(column->view(0, threadSize));
        cpu2 = new DBEngineCPU(column->view(threadSize, threadSize*2));
        cpu3 = new DBEngineCPU(column->view(threadSize*2, threadSize*3));
        cpu4 = new DBEngineCPU(column->view(threadSize*3, threadSize*4));
     }

    virtual int filterCount(int minIncluded, int maxExcluded) {
       std::future<int> fut1 = std::async(&DBEngineCPU::filterCount, cpu1, minIncluded, maxExcluded);
       std::future<int> fut2 = std::async(&DBEngineCPU::filterCount, cpu2, minIncluded, maxExcluded);
       std::future<int> fut3 = std::async(&DBEngineCPU::filterCount, cpu3, minIncluded, maxExcluded);
       std::future<int> fut4 = std::async(&DBEngineCPU::filterCount, cpu4, minIncluded, maxExcluded);

       return (fut1.get() + fut2.get() + fut3.get() + fut4.get());
    }

    virtual double average() {
        std::future<double> fut1 = std::async(&DBEngineCPU::average, cpu1);
        std::future<double> fut2 = std::async(&DBEngineCPU::average, cpu2);
        std::future<double> fut3 = std::async(&DBEngineCPU::average, cpu3);
        std::future<double> fut4 = std::async(&DBEngineCPU::average, cpu4);

        return (fut1.get() + fut2.get() + fut3.get() + fut4.get()) / 4;
    }

    virtual int min() {
        std::future<int> fut1 = std::async(&DBEngineCPU::min, cpu1);
        std::future<int> fut2 = std::async(&DBEngineCPU::min, cpu2);
        std::future<int> fut3 = std::async(&DBEngineCPU::min, cpu3);
        std::future<int> fut4 = std::async(&DBEngineCPU::min, cpu4);

        return std::min(std::min(fut1.get(), fut2.get()), std::min(fut3.get(), fut4.get()));
    }

    virtual int max() {
        std::future<int> fut1 = std::async(&DBEngineCPU::max, cpu1);
        std::future<int> fut2 = std::async(&DBEngineCPU::max, cpu2);
        std::future<int> fut3 = std::async(&DBEngineCPU::max, cpu3);
        std::future<int> fut4 = std::async(&DBEngineCPU::max, cpu4);

        return std::max(std::max(fut1.get(), fut2.get()), std::max(fut3.get(), fut4.get()));
    }

    virtual const char* name() {
      return "CPU 4";
    }
};

class DBEngineNPP: public DBEngine {
  protected:
    int *deviceData;
    bool pinned;
    bool writeCombining;

    static Npp8u* alloc8u(int numElements) {
      Npp8u* pBuffer; // Pointer for the buffer
      cudaMalloc(&pBuffer, numElements);

      return pBuffer;
    }

    static Npp32s* alloc32s(int numElements) {
      Npp32s* pBuffer; // Pointer for the buffer
      cudaMalloc(&pBuffer, sizeof(Npp32s) * numElements);

      return pBuffer;
    }

  public:
    DBEngineNPP(DBColumn *column, bool pinMemory, bool useWriteCombining): DBEngine(column) { deviceData = NULL; pinned = pinMemory; writeCombining = useWriteCombining; }

    virtual int filterCount(int minIncluded, int maxExcluded) {
        Npp32s bufferSize, countHost;
        nppsCountInRangeGetBufferSize_32s(col->size, &bufferSize);
        Npp8u* pBuffer = alloc8u(bufferSize);
        Npp32s* pCountDevice = alloc32s(1);

        nppsCountInRange_32s(deviceData, col->size, pCountDevice, minIncluded, maxExcluded + 1, pBuffer);
        cudaMemcpy(&countHost, pCountDevice, sizeof(Npp32s), cudaMemcpyDeviceToHost);

        cudaFree(pBuffer);

        return countHost;
    }

    virtual double average() {
        Npp32s bufferSize, meanHost;
        nppsMeanGetBufferSize_32s_Sfs(col->size, &bufferSize);
        Npp8u* pBuffer = alloc8u(bufferSize);
        Npp32s* pMeanDevice = alloc32s(1);

        nppsMean_32s_Sfs(deviceData, col->size, pMeanDevice, 0, pBuffer);
        cudaMemcpy(&meanHost, pMeanDevice, sizeof(Npp32s), cudaMemcpyDeviceToHost);

        cudaFree(pBuffer);

        return meanHost;
    }

    virtual int min() {
        Npp32s bufferSize, minHost;
        nppsMinGetBufferSize_32s(col->size, &bufferSize);
        Npp8u* pBuffer = alloc8u(bufferSize);
        Npp32s* pMinDevice = alloc32s(1);

        nppsMin_32s(deviceData, col->size, pMinDevice, pBuffer);
        cudaMemcpy(&minHost, pMinDevice, sizeof(Npp32s), cudaMemcpyDeviceToHost);

        cudaFree(pBuffer);

        return minHost;
    }

    virtual int max() {
        Npp32s bufferSize, maxHost;
        nppsMaxGetBufferSize_32s(col->size, &bufferSize);
        Npp8u* pBuffer = alloc8u(bufferSize);
        Npp32s* pMaxDevice = alloc32s(1);

        nppsMax_32s(deviceData, col->size, pMaxDevice, pBuffer);
        cudaMemcpy(&maxHost, pMaxDevice, sizeof(Npp32s), cudaMemcpyDeviceToHost);

        cudaFree(pBuffer);

        return maxHost;
    }

    virtual void init() {
        assert(deviceData == NULL);
        if (writeCombining)
          assert(pinned);

        else if (pinned)
            cudaHostAlloc(&deviceData, sizeof(int) * col->size, writeCombining ? cudaHostAllocWriteCombined : 0);
        else
            cudaMalloc(&deviceData, sizeof(int) * col->size);
    }

    virtual void copy() {
        cudaMemcpy(deviceData, col->data, sizeof(int) * col->size, cudaMemcpyHostToDevice);
    }

    virtual const char* name() {
      return writeCombining ? "NPP Write combining" : (pinned ? "NPP - pinned" : "NPP");
    }
};

class DBEngineNPPInit: public DBEngineNPP {
    public:
        DBEngineNPPInit(DBColumn *column, bool pinMemory, bool useWriteCombining) : DBEngineNPP(column, pinMemory, useWriteCombining) { }

        virtual const char* name() {
          return writeCombining ? "NPP INIT Write combining" : (pinned ? "NPP INIT - pinned" : "NPP INIT");
        }
};

void runTest(DBEngine *engine) {
    auto start = std::chrono::high_resolution_clock::now();

    engine->init();
    auto init = std::chrono::high_resolution_clock::now();
    engine->copy();
    auto copy = std::chrono::high_resolution_clock::now();

    int count = engine->filterCount(-10000, 10000);
    double average = engine->average();
    int min = engine->min();
    int max = engine->max();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto durationInit = std::chrono::duration_cast<std::chrono::milliseconds>(init - start);
    auto durationCopy = std::chrono::duration_cast<std::chrono::milliseconds>(copy - init);
    std::cout << engine->name() << " - Time: " << duration.count() << " ms - Init: " << durationInit.count() << " ms" << " - Copy: " << durationCopy.count() << " - Count: " << count << " - Average: " << average << " - Min: " << min << " - Max: " << max << std::endl;
}

int main(int argc, char** argv) {
    printf("Submission of Luca Venturi\n");

    DBColumn col(100000000);
    printf("Randomizing\n");
    col.randomize();

    printf("Creating engines\n");
    DBEngineCPU cpu(&col);
    DBEngineNPP nppPinned(&col, true, false);
    DBEngineNPP npp(&col, false, false);
    DBEngineCPU4Threads cpu4(&col);
    // Used to start cuda libraries
    DBEngineNPPInit nppInit(&col, false, false);

    runTest(&cpu);
    runTest(&cpu4);
    runTest(&nppInit);
    runTest(&npp);
    runTest(&nppPinned);
}