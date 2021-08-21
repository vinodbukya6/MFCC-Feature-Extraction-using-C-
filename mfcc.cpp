#include<algorithm>
#include<numeric>
#include<complex>
#include<vector>
#include<map>
#include<math.h>
#include"wavHeader.h"
#include <iostream>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <string>
#include<stdio.h>
#include<sys/stat.h>
#include<string.h>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <list> // for list operations
#include <jni.h>
#include <android/log.h>
#include <errno.h>

typedef std::vector<double> v_d;
typedef std::complex<double> c_d;
typedef std::vector<v_d> m_d;
typedef std::vector<c_d> v_c_d;
typedef std::map<int,std::map<int,c_d> > twmap;

std::string vectorS;
std::string vectorMfccCoef;
std::vector<double> vect;
std::string str3;
std::ofstream myfile;
std::ofstream outMeanFile;

class MFCC {

private:
    const double PI = 4*atan(1.0);   // Pi = 3.14...
    int fs;
    twmap twiddle;
    size_t winLengthSamples, frameShiftSamples, numCepstra, numFFT, numFFTBins, numFilters;
    double preEmphCoef, lowFreq, highFreq;
    v_d frame, powerSpectralCoef, lmfbCoef, hamming, mfcc, prevsamples;
    m_d fbank, dct;

private:
    // Hertz to Mel conversion
    inline double hz2mel (double f) {
        return 2595*std::log10 (1+f/700);
    }

    // Mel to Hertz conversion
    inline double mel2hz (double m) {
        return 700*(std::pow(10,m/2595)-1);
    }

    // Twiddle factor computation
    void compTwiddle(void) {
        const c_d J(0,1);      // Imaginary number 'j'
        for (int N=2; N<=numFFT; N*=2)
            for (int k=0; k<=N/2-1; k++)
                twiddle[N][k] = exp(-2*PI*k/N*J);
    }

    // Cooley-Tukey DIT-FFT recursive function
    v_c_d fft(v_c_d x) {
        int N = x.size();
        if (N==1)
            return x;

        v_c_d xe(N/2,0), xo(N/2,0), Xjo, Xjo2;
        int i;

        // Construct arrays from even and odd indices
        for (i=0; i<N; i+=2)
            xe[i/2] = x[i];
        for (i=1; i<N; i+=2)
            xo[(i-1)/2] = x[i];

        // Compute N/2-point FFT
        Xjo = fft(xe);
        Xjo2 = fft(xo);
        Xjo.insert (Xjo.end(), Xjo2.begin(), Xjo2.end());

        // Butterfly computations
        for (i=0; i<=N/2-1; i++) {
            c_d t = Xjo[i], tw = twiddle[N][i];
            Xjo[i] = t + tw * Xjo[i+N/2];
            Xjo[i+N/2] = t - tw * Xjo[i+N/2];
        }
        return Xjo;
    }

    //// Frame processing routines
    // Pre-emphasis and Hamming window
    void preEmphHam(void) {
        v_d procFrame(frame.size(), hamming[0]*frame[0]);
        for (int i=1; i<frame.size(); i++)
            procFrame[i] = hamming[i] * (frame[i] - preEmphCoef * frame[i-1]);
        frame = procFrame;
    }

    // Power spectrum computation
    void computePowerSpec(void) {
        frame.resize(numFFT); // Pads zeros
        v_c_d framec (frame.begin(), frame.end()); // Complex frame
        v_c_d fftc = fft(framec);

        for (int i=0; i<numFFTBins; i++)
            powerSpectralCoef[i] = pow(abs(fftc[i]),2);
    }

    // Applying log Mel filterbank (LMFB)
    void applyLMFB(void) {
        lmfbCoef.assign(numFilters,0);

        for (int i=0; i<numFilters; i++) {
            // Multiply the filterbank matrix
            for (int j=0; j<fbank[i].size(); j++)
                lmfbCoef[i] += fbank[i][j] * powerSpectralCoef[j];
            // Apply Mel-flooring
            if (lmfbCoef[i] < 1.0)
                lmfbCoef[i] = 1.0;
        }

        // Applying log on amplitude
        for (int i=0; i<numFilters; i++)
            lmfbCoef[i] = std::log (lmfbCoef[i]);
    }

    // Computing discrete cosine transform
    void applyDct(void) {
        mfcc.assign(numCepstra+1,0);
        for (int i=0; i<=numCepstra; i++) {
            for (int j=0; j<numFilters; j++)
                mfcc[i] += dct[i][j] * lmfbCoef[j];
        }
    }

    // Initialisation routines
    // Pre-computing Hamming window and dct matrix
    void initHamDct(void) {
        int i, j;

        hamming.assign(winLengthSamples,0);
        for (i=0; i<winLengthSamples; i++)
            hamming[i] = 0.54 - 0.46 * cos(2 * PI * i / (winLengthSamples-1));

        v_d v1(numCepstra+1,0), v2(numFilters,0);
        for (i=0; i <= numCepstra; i++)
            v1[i] = i;
        for (i=0; i < numFilters; i++)
            v2[i] = i + 0.5;

        dct.reserve (numFilters*(numCepstra+1));
        double c = sqrt(2.0/numFilters);
        for (i=0; i<=numCepstra; i++) {
            v_d dtemp;
            for (j=0; j<numFilters; j++)
                dtemp.push_back (c * cos(PI / numFilters * v1[i] * v2[j]));
            dct.push_back(dtemp);
        }
    }

    // Precompute filterbank
    void initFilterbank () {
        // Convert low and high frequencies to Mel scale
        double lowFreqMel = hz2mel(lowFreq);
        double highFreqMel = hz2mel (highFreq);

        // Calculate filter centre-frequencies
        v_d filterCentreFreq;
        filterCentreFreq.reserve (numFilters+2);
        for (int i=0; i<numFilters+2; i++)
            filterCentreFreq.push_back (mel2hz(lowFreqMel + (highFreqMel-lowFreqMel)/(numFilters+1)*i));

        // Calculate FFT bin frequencies
        v_d fftBinFreq;
        fftBinFreq.reserve(numFFTBins);
        for (int i=0; i<numFFTBins; i++)
            fftBinFreq.push_back (fs/2.0/(numFFTBins-1)*i);

        // Filterbank: Allocate memory
        fbank.reserve (numFilters*numFFTBins);

        // Populate the fbank matrix
        for (int filt=1; filt<=numFilters; filt++) {
            v_d ftemp;
            for (int bin=0; bin<numFFTBins; bin++) {
                double weight;
                if (fftBinFreq[bin] < filterCentreFreq[filt-1])
                    weight = 0;
                else if (fftBinFreq[bin] <= filterCentreFreq[filt])
                    weight = (fftBinFreq[bin] - filterCentreFreq[filt-1]) / (filterCentreFreq[filt] - filterCentreFreq[filt-1]);
                else if (fftBinFreq[bin] <= filterCentreFreq[filt+1])
                    weight = (filterCentreFreq[filt+1] - fftBinFreq[bin]) / (filterCentreFreq[filt+1] - filterCentreFreq[filt]);
                else
                    weight = 0;
                ftemp.push_back (weight);
            }
            fbank.push_back(ftemp);
        }
    }

    // Convert vector of double to string (for writing MFCC file output)
    std::string v_d_to_string (v_d vec) {
        std::stringstream vecStream;
        for (int i=0; i<vec.size()-1; i++) {
            vecStream << std::scientific << vec[i];
            vecStream << ",";
        }
        vecStream << std::scientific << vec.back();
        vecStream << ",";
        return vecStream.str();
    }

public:
    // MFCC class constructor
    MFCC(int sampFreq=16000, int nCep=12, int winLength=25, int frameShift=10, int numFilt=40, double lf=50, double hf=6500) {
        fs          = sampFreq;             // Sampling frequency
        numCepstra  = nCep;                 // Number of cepstra
        numFilters  = numFilt;              // Number of Mel warped filters
        preEmphCoef = 0.97;                 // Pre-emphasis coefficient
        lowFreq     = lf;                   // Filterbank low frequency cutoff in Hertz
        highFreq    = hf;                   // Filterbank high frequency cutoff in Hertz
        numFFT      = fs<=20000?512:2048;   // FFT size
        winLengthSamples   = winLength * fs / 1e3;  // winLength in milliseconds
        frameShiftSamples  = frameShift * fs / 1e3; // frameShift in milliseconds

        numFFTBins = numFFT/2 + 1;
        powerSpectralCoef.assign (numFFTBins, 0);
        prevsamples.assign (winLengthSamples-frameShiftSamples, 0);

        initFilterbank();
        initHamDct();
        compTwiddle();
    }

    // Process each frame and extract MFCC
    std::string processFrame(int16_t* samples, size_t N) {
        // Add samples from the previous frame that overlap with the current frame
        // to the current samples and create the frame.
        frame = prevsamples;
        for (int i=0; i<N; i++)
            frame.push_back(samples[i]);
        prevsamples.assign(frame.begin()+frameShiftSamples, frame.end());

        preEmphHam();
        computePowerSpec();
        applyLMFB();
        applyDct();

        return v_d_to_string (mfcc);
    }

    // Read input file stream, extract MFCCs and write to output file stream
    int process (std::ifstream &wavFp) {
        // Read the wav header
        wavHeader hdr;
        int headerSize = sizeof(wavHeader);
        wavFp.read((char *) &hdr, headerSize);

        // Check audio format
        if (hdr.AudioFormat != 1 || hdr.bitsPerSample != 16) {
            std::cerr << "Unsupported audio format, use 16 bit PCM Wave" << std::endl;
            return 1;
        }
        // Check sampling rate
        if (hdr.SamplesPerSec != 16000) {
            std::cerr << "Sampling rate mismatch: Found " << hdr.SamplesPerSec << " instead of " << fs <<std::endl;
            return 1;
        }

        // Check sampling rate
        if (hdr.NumOfChan != 1) {
            std::cerr << hdr.NumOfChan << " channel files are unsupported. Use mono." <<std::endl;
            return 1;
        }

        // Initialise buffer
        uint16_t bufferLength = winLengthSamples-frameShiftSamples;
        int16_t* buffer = new int16_t[bufferLength];
        int bufferBPS = (sizeof buffer[0]);

        // Read and set the initial samples
        wavFp.read((char *) buffer, bufferLength*bufferBPS);
        for (int i=0; i<bufferLength; i++)
            prevsamples[i] = buffer[i];
        delete [] buffer;

        // Recalculate buffer size
        bufferLength = frameShiftSamples;
        buffer = new int16_t[bufferLength];

        vectorS = "";
        vectorMfccCoef = "";
        //    vectorMfccCoef.append(str3 + "=");

        // Read data and process each frame
        wavFp.read((char *) buffer, bufferLength*bufferBPS);
        while (wavFp.gcount() == bufferLength*bufferBPS && !wavFp.eof()) {
            vectorS.append(processFrame(buffer, bufferLength));
            wavFp.read((char *) buffer, bufferLength*bufferBPS);
        }

        // Determine which MFCC coefficient to compute
        unsigned int coeff;

        // Holds the value of the computed coefficient
        double mfcc_result;

        // Compute the first 13 coefficients
        /*for(coeff = 0; coeff < 40; coeff++)
        {
            mfcc_result = GetCoefficient(16000, 40, 25, coeff);
            //      printf("%i %f\n", coeff, mfcc_result);
            //     vectorMfccCoef.append(std::to_string(mfcc_result) + ",");
        }*/

        totalMfccStr();

        delete [] buffer;
        buffer = nullptr;
        return 0;
    }

    // Return mfcc numbers array in string
    std::string totalMfccStr() {
        __android_log_print(ANDROID_LOG_ERROR, "DIR",  "Mean Value. %d\n", vectorMfccCoef.c_str());
        return vectorMfccCoef;
    }


    //double* spectralData,
    double GetCoefficient(unsigned int samplingRate, unsigned int NumFilters, unsigned int binSize, unsigned int m)
    {
        double result = 0.0f;
        double outerSum = 0.0f;
        double innerSum = 0.0f;
        unsigned int k, l;

        // 0 <= m < L
        if(m >= NumFilters)
        {
            // This represents an error condition - the specified coefficient is greater than or equal to the number of filters. The behavior in this case is undefined.
            return 0.0f;
        }

        result = NormalizationFactor(NumFilters, m);


        for(l = 1; l <= NumFilters; l++)
        {
            // Compute inner sum
            innerSum = 0.0f;
            for(k = 0; k < binSize - 1; k++)
            {
                innerSum += fabs(powerSpectralCoef[k] * GetFilterParameter(samplingRate, binSize, k, l));
            }

            if(innerSum > 0.0f)
            {
                innerSum = log(innerSum); // The log of 0 is undefined, so don't use it
            }

            innerSum = innerSum * cos(((m * PI) / NumFilters) * (l - 0.5f));

            outerSum += innerSum;
        }

        result *= outerSum;

        return result;
    }

    /*
     * Computes the Normalization Factor (Equation 6)
     * Used for internal computation only - not to be called directly
     */
    double NormalizationFactor(int NumFilters, int m)
    {
        double normalizationFactor = 0.0f;

        if(m == 0)
        {
            normalizationFactor = sqrt(1.0f / NumFilters);
        }
        else
        {
            normalizationFactor = sqrt(2.0f / NumFilters);
        }

        return normalizationFactor;
    }

    /*
     * Compute the filter parameter for the specified frequency and filter bands (Eq. 2)
     * Used for internal computation only - not the be called directly
     */
    double GetFilterParameter(unsigned int samplingRate, unsigned int binSize, unsigned int frequencyBand, unsigned int filterBand)
    {
        double filterParameter = 0.0f;

        double boundary = (frequencyBand * samplingRate) / binSize;        // k * Fs / N
        double prevCenterFrequency = GetCenterFrequency(filterBand - 1);        // fc(l - 1) etc.
        double thisCenterFrequency = GetCenterFrequency(filterBand);
        double nextCenterFrequency = GetCenterFrequency(filterBand + 1);

        if(boundary >= 0 && boundary < prevCenterFrequency)
        {
            filterParameter = 0.0f;
        }
        else if(boundary >= prevCenterFrequency && boundary < thisCenterFrequency)
        {
            filterParameter = (boundary - prevCenterFrequency) / (thisCenterFrequency - prevCenterFrequency);
            filterParameter *= GetMagnitudeFactor(filterBand);
        }
        else if(boundary >= thisCenterFrequency && boundary < nextCenterFrequency)
        {
            filterParameter = (boundary - nextCenterFrequency) / (thisCenterFrequency - nextCenterFrequency);
            filterParameter *= GetMagnitudeFactor(filterBand);
        }
        else if(boundary >= nextCenterFrequency && boundary < samplingRate)
        {
            filterParameter = 0.0f;
        }

        return filterParameter;
    }

    /*
     * Compute the band-dependent magnitude factor for the given filter band (Eq. 3)
     * Used for internal computation only - not the be called directly
     */
    double GetMagnitudeFactor(unsigned int filterBand)
    {
        double magnitudeFactor = 0.0f;

        if(filterBand >= 1 && filterBand <= 14)
        {
            magnitudeFactor = 0.015;
        }
        else if(filterBand >= 15 && filterBand <= 48)
        {
            magnitudeFactor = 2.0f / (GetCenterFrequency(filterBand + 1) - GetCenterFrequency(filterBand -1));
        }

        return magnitudeFactor;
    }

    /*
     * Compute the center frequency (fc) of the specified filter band (l) (Eq. 4)
     * This where the mel-frequency scaling occurs. Filters are specified so that their
     * center frequencies are equally spaced on the mel scale
     * Used for internal computation only - not the be called directly
     */
    double GetCenterFrequency(unsigned int filterBand)
    {
        double centerFrequency = 0.0f;
        double exponent;

        if(filterBand == 0)
        {
            centerFrequency = 0;
        }
        else if(filterBand >= 1 && filterBand <= 14)
        {
            centerFrequency = (200.0f * filterBand) / 3.0f;
        }
        else
        {
            exponent = filterBand - 14.0f;
            centerFrequency = pow(1.0711703, exponent);
            centerFrequency *= 1073.4;
        }

        return centerFrequency;
    }

};

// Compute-mfcc function

// A simple option parser
char* getCmdOption(char **begin, char **end, const std::string &value) {
    char **iter = std::find(begin, end, value);
    if (iter != end && ++iter != end)
        return *iter;
    return nullptr;
}

// Process each file
int processFile (MFCC &mfccComputer, const char* wavPath) {
    // Initialise input and output streams
    std::ifstream wavFp;
    std::ofstream mfcFp;

    // Check if input is readable
    wavFp.open(wavPath);
    if (!wavFp.is_open()) {
        std::cerr << "Unable to open input file: " << wavPath << std::endl;
        return 1;
    }

    // Extract and write features
    if (mfccComputer.process (wavFp))
        std::cerr << "Error processing " << wavPath << std::endl;

    wavFp.close();
    mfcFp.close();
    return 0;
}

// Process lists
int processList (MFCC &mfccComputer, const char* wavListPath, const char* mfcListPath) {
    std::ifstream wavListFp, mfcListFp;

    // Check if wav list is readable
    wavListFp.open(wavListPath);
    if (!wavListFp.is_open()) {
        std::cerr << "Unable to open input list: " << wavListPath << std::endl;
        return 1;
    }

    // Check if mfc list is readable
    mfcListFp.open(mfcListPath);
    if (!mfcListFp.is_open()) {
        std::cerr << "Unable to open output list: " << mfcListPath << std::endl;
        return 1;
    }

    // Process lists
    std::string wavPath, mfcPath;
    while (true) {
        std::getline (wavListFp, wavPath);
        std::getline (mfcListFp, mfcPath);
        if (wavPath.empty() || mfcPath.empty()) {
            wavListFp.close();
            mfcListFp.close();
            return 0;
        }
        if (processFile (mfccComputer, wavPath.c_str())) {
            wavListFp.close();
            mfcListFp.close();
            return 1;
        }
    }
}

// Main pass wav file path
int inputCheckParm(const char* wavPath) {

    // Check arguments
    if (!(wavPath)) {
        return 1;
    }

    // Assign variables
    int numCepstra = 13;
    int numFilters = 40;
    int samplingRate = 16000;
    int winLength = 25;
    int frameShift = 10;
    int lowFreq = 50;
    int highFreq = 6500;

    // Initialise MFCC class instance
    MFCC mfccComputer (samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);

    // Process wav files
    if (wavPath)
        if (processFile (mfccComputer, wavPath))
            return 1;

    return 0;
}

// final main for extracting feature frame levels and mean them
int mainToExtractMFCC(const char* wavPath){

    inputCheckParm(wavPath);

    //  int number_of_lines = 0;
    std::string line;
    //  double xFrames[5000][14] = {};
    std::vector<double> vect;
    std::stringstream ss(vectorS);
    for (double i; ss >> i;) {
        vect.push_back(i);
        if (ss.peek() == ',')
            ss.ignore();
    }
    int countFrames = vect.size()/14;
    double xFrames[5000][14] = {};
    int pos = 0;
    for(int i=0;i<countFrames;i++){
        for(int j=0;j<14;j++) {
            xFrames[i][j]= vect[pos];
            //    std::cout <<std::fixed<<xFrames[j][i]<<std::endl;
            pos++;
        }
    }

    std::cout<<xFrames<<std::endl;
    std::cout<<"..done.."<<std::endl;
    double yMean[14] = { };

    // output each array element's value
    for (int i = 0; i < 14; i++)
    {
        double sum=0;
        for (int j = 0; j <countFrames; j++)
        {
            sum=sum+xFrames[j][i];
            //      std::cout <<std::fixed<<xFrames[j][i]<<std::endl;
            //    std::cout<<sum<<std::endl;
        }

        yMean[i]=sum/countFrames;
    }
    for (int k=0; k<14; k++)
    {
        //  outMeanFile<<yMean[k]<<",";
        //  std::cout<<yMean[k]<<std::endl;
        vectorMfccCoef.append(std::to_string(yMean[k]) + ",");
        std::cout<<vectorMfccCoef<<std::endl;
    }

    return 0;
}
