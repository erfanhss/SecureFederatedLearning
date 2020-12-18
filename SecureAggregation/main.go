package main

import (
	//"github.com/ldsec/lattigo/v2/bfv"
	//"github.com/ldsec/lattigo/v2/bfv"
	//"github.com/ldsec/lattigo/v2/dbfv"
	"github.com/ldsec/lattigo/v2/ckks"
	"C"
	//"github.com/ldsec/lattigo/v2/dbfv"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
	"math/rand"
	"time"
)


func Pow(x uint64, y uint64, p uint64) uint64 {
	if y == 0 {
		return 1
	} else {
		var res uint64 = 1
		var i uint64 = 1
		for i = 0; i < y; i++ {
			res = MultiplyNonOverflow(res, x, p)
		}
		return res
	}
}


func MultiplyNonOverflow(x uint64, y uint64, p uint64) uint64 {
	locA := x
	locB := y
	if locB == 1 {
		return locA
	}
	if locA == 1 {
		return locB
	}
	if locB%2 == 0 {
		locA = (2 * locA) % p
		locB = locB / 2
		return MultiplyNonOverflow(locA, locB, p)
	} else {
		return (locA + MultiplyNonOverflow(locA, locB-1, p)) % p
	}

}

func MultiplicativeInverse(x uint64, p uint64) uint64 {
	r_0 := int64(p)
	r_1 := int64(x)
	t_0 := int64(0)
	t_1 := int64(1)
	for {
		q := r_0 / r_1
		new_r := r_0 - q*r_1
		if new_r == 0 {
			break
		}
		new_t := t_0 - q*t_1
		r_0 = r_1
		t_0 = t_1
		r_1 = new_r
		t_1 = new_t
	}
	if t_1 > 0 {
		return uint64(t_1)
	} else {
		return uint64(int64(p) + t_1)
	}
}

func GenerateVandermonde(evalPoint uint64, threshold uint64, p uint64) []uint64 {
	res := make([]uint64, threshold)
	var i uint64 = 0
	for i = 0; i < threshold; i++ {
		res[i] = Pow(evalPoint, i, p)
	}
	return res
}

func SubField(a uint64, b uint64, p uint64) uint64 {
	if a > b {
		return (a - b) % p
	} else {
		return SubField(a+p, b, p)
	}
}

func GenerateVandermondeInverse(evalPoints []uint64, p uint64) []uint64 {
	res := make([]uint64, len(evalPoints))
	var i int
	for i = 0; i < len(evalPoints); i++ {
		var innerRes uint64 = 1
		for m := 0; m < len(evalPoints); m++ {
			if m != i {
				tmp := MultiplyNonOverflow(evalPoints[m], MultiplicativeInverse(SubField(evalPoints[m], evalPoints[i], p), p), p)
				innerRes = MultiplyNonOverflow(tmp, innerRes, p)
			}
		}
		res[i] = innerRes
	}
	return res
}
func check(err error) {
	if err != nil {
		panic(err)
	}
}

func runTimed(f func()) time.Duration {
	start := time.Now()
	f()
	return time.Since(start)
}

func runTimedParty(f func(), N int) time.Duration {
	start := time.Now()
	f()
	return time.Duration(time.Since(start).Nanoseconds() / int64(N))
}

//export SecureAggregation
func SecureAggregation(inputs []float64, N int, k int, logDegree uint64, res []float64) (float64, float64) {
	var ringPrime uint64 = 0xfffffffffffc001
	numElements := len(inputs) / N
	//l := log.New(os.Stderr, "", 0)
	type party struct {
		//sk           []*bfv.SecretKey
		sk 				[]*ckks.SecretKey
		secretShares []*ring.Poly
		shamirShare  *ring.Poly
		ckgShare 	 dckks.CKGShare
		//ckgShare     dbfv.CKGShare
		//pcksShare    dbfv.PCKSShare
		pcksShare 	 dckks.PCKSShare
		//input 		 []uint64
		input        []complex128
	}

	//moduli := &bfv.Moduli{[]uint64{ringPrime}, []uint64{ringPrime}, []uint64{ringPrime}}
	moduli := &ckks.Moduli{[]uint64{ringPrime}, []uint64{ringPrime}}
	//params, err := bfv.NewParametersFromModuli(logDegree, moduli, 65537)
	params, err := ckks.NewParametersFromModuli(logDegree, moduli)
	params.SetScale(1<<20)
	params.SetLogSlots(logDegree-1)
	lattigoPRNG, err := utils.NewKeyedPRNG([]byte{'l', 'a', 't', 't', 'i', 'g', 'o'})
	if err != nil {
		panic(err)
	}
	// Ring for the common reference polynomials sampling
	ringQP, _ := ring.NewRing(1<<params.LogN(), append(params.Qi(), params.Pi()...))
	ringQ, _ := ring.NewRing(1<<params.LogN(), params.Qi())
	// Common reference polynomial generator that uses the PRNG
	crsGen := ring.NewUniformSampler(lattigoPRNG, ringQP)
	crs := crsGen.ReadNew() // for the public-key

	// Target private and public keys
	//tsk, tpk := bfv.NewKeyGenerator(params).GenKeyPair()
	tsk, tpk := ckks.NewKeyGenerator(params).GenKeyPair()

	//ckg := dbfv.NewCKGProtocol(params)
	ckg := dckks.NewCKGProtocol(params)
	//pcks := dbfv.NewPCKSProtocol(params, 3.19)
	pcks := dckks.NewPCKSProtocol(params, 3.19)
	P := make([]*party, N, N)
	evalPoints := make([]uint64, N)
	for i := range P {
		rand.Seed(time.Now().UTC().UnixNano())
		evalPoints[i] = rand.Uint64() % ringPrime
	}
	for i := range P {
		pi := &party{}
		//pi.sk = make([]*bfv.SecretKey, k, k)
		pi.sk = make([]*ckks.SecretKey, k, k)
		pi.secretShares = make([]*ring.Poly, N, N)
		// create k different secret keys for each party
		for partyCntr := 0; partyCntr < k; partyCntr++ {
			//pi.sk[partyCntr] = bfv.NewKeyGenerator(params).GenSecretKey()
			pi.sk[partyCntr] = ckks.NewKeyGenerator(params).GenSecretKey()
		}
		// create the shares of the secret key
		for partyCntr := range P {
			vandermonde := GenerateVandermonde(evalPoints[partyCntr], uint64(k), ringPrime)
			res := ringQ.NewPoly()
			ringQ.MulScalar(pi.sk[0].Get(), vandermonde[0], res)
			for i := 1; i < k; i++ {
				tmp := ringQ.NewPoly()
				ringQ.MulScalar(pi.sk[i].Get(), vandermonde[i], tmp)
				ringQ.Add(tmp, res, res)
			}
			pi.secretShares[partyCntr] = res
		}
		// Create each party, and allocate the memory for all the shares that the protocols will need
		// creating inputs
		pi.input = make([]complex128, params.Slots(), params.Slots())
		for j:=0;j<numElements;j++{
			pi.input[j] = complex(inputs[i*numElements + j], 0)
		}

		pi.ckgShare = ckg.AllocateShares()
		pi.pcksShare = pcks.AllocateShares(params.MaxLevel())
		P[i] = pi
	}

	// generate shamir shares
	for i := range P {
		res := ringQ.NewPoly()
		ringQ.Add(P[0].secretShares[i], P[1].secretShares[i], res)
		for j := 2; j < N; j++ {
			ringQ.Add(res, P[j].secretShares[i], res)
		}
		P[i].shamirShare = res
	}

	var elapsedCKGCloud time.Duration
	var elapsedCKGParty time.Duration
	var elapsedRKGCloud time.Duration
	var elapsedRKGParty time.Duration

	// 1) Collective public key generation
	//pk := bfv.NewPublicKey(params)
	pk := ckks.NewPublicKey(params)
	elapsedCKGParty = runTimedParty(func() {
		for _, pi := range P {
			ckg.GenShare(pi.sk[0].Get(), crs, pi.ckgShare)
		}
	}, N)
	ckgCombined := ckg.AllocateShares()
	elapsedCKGCloud = runTimed(func() {
		for _, pi := range P {
			ckg.AggregateShares(pi.ckgShare, ckgCombined, ckgCombined)
		}
		ckg.GenPublicKey(ckgCombined, crs, pk)
	})
	// Pre-loading memory
	//encInputs := make([]*bfv.Ciphertext, N, N)
	encInputs := make([]*ckks.Ciphertext, N, N)
	for i := range encInputs {
		//encInputs[i] = bfv.NewCiphertext(params, 1)
		encInputs[i] = ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
	}
	//encResult := bfv.NewCiphertext(params, 1)
	encResult := ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
	// Each party encrypts its input vector
	//encryptor := bfv.NewEncryptorFromPk(params, pk)
	//encoder := bfv.NewEncoder(params)
	//pt := bfv.NewPlaintext(params)
	encryptor := ckks.NewEncryptorFromPk(params, pk)
	encoder := ckks.NewEncoder(params)
	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())

	elapsedEncryptParty := runTimedParty(func() {
		for i, pi := range P {
			//encoder.EncodeUint(pi.input, pt)
			encoder.Encode(pt, pi.input, params.Slots())
			encryptor.Encrypt(pt, encInputs[i])
		}
	}, N)

	elapsedEncryptCloud := time.Duration(0)
	//evaluator := bfv.NewEvaluator(params)
	evaluator := ckks.NewEvaluator(params)
	elapsedEvalCloudCPU := runTimed(func() {
		evaluator.Add(encInputs[0], encInputs[1], encResult)
		for i := 2; i < N; i++ {
			evaluator.Add(encResult, encInputs[i], encResult)
		}
	})

	elapsedEvalParty := time.Duration(0)
	// Collective key switching from the collective secret key to
	// the target public key
	elapsedPCKSParty := runTimedParty(func() {
		inverse := GenerateVandermondeInverse(evalPoints[0:k], ringPrime)
		for i := 0; i < k; i++ {
			scaledSecretKey := ringQ.NewPoly()
			ringQ.MulScalar(P[i].sk[0].Get(), inverse[i], scaledSecretKey)
			pcks.GenShare(scaledSecretKey, tpk, encResult, P[i].pcksShare)
		}

	}, N)

	pcksCombined := pcks.AllocateShares(params.MaxLevel())
	//encOut := bfv.NewCiphertext(params, 1)
	encOut := ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
	elapsedPCKSCloud := runTimed(func() {
		for i := 0; i < k; i++ {
			pcks.AggregateShares(P[i].pcksShare, pcksCombined, pcksCombined)
		}
		pcks.KeySwitch(pcksCombined, encResult, encOut)

	})
	// Decrypt the result with the target secret key
	//decryptor := bfv.NewDecryptor(params, tsk)
	//ptres := bfv.NewPlaintext(params)

	decryptor := ckks.NewDecryptor(params, tsk)
	ptres := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())

	elapsedDecParty := runTimed(func() {
		decryptor.Decrypt(encOut, ptres)
	})
	// Check the result
	//tmp := encoder.DecodeUint(ptres)[0:numElements]
	tmp := encoder.Decode(ptres, params.Slots())
	for i:=0;i<numElements;i++{
		res[i] = real(tmp[i])
	}
	timeCloud := elapsedCKGCloud.Seconds() + elapsedRKGCloud.Seconds() + elapsedEncryptCloud.Seconds() + elapsedEvalCloudCPU.Seconds() + elapsedPCKSCloud.Seconds()
	timeParty := elapsedCKGParty.Seconds() + elapsedRKGParty.Seconds() + elapsedEncryptParty.Seconds() + elapsedEvalParty.Seconds() + elapsedPCKSParty.Seconds() + elapsedDecParty.Seconds()
	return timeCloud, timeParty

}

func main() {
	//numElements := 2
	//N := 30
	//k := 2
	//var inputs []float64
	//inputs = make([]float64, N*numElements, N*numElements)
	//for i := 0; i < N*numElements; i++ {
	//	rand.Seed(time.Now().UnixNano())
	//	inputs[i] = rand.Float64()*3
	//}
	//log.Println(inputs)
	//var res []float64
	//res = make([]float64, numElements, numElements)
	//log.Println(SecureAggregation(inputs, N, k, 13, res))
	//log.Println(res)
}
