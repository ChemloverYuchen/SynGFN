import sys

from omegaconf import ListConfig
from potts_utils import load_potts_model, potts_energy
from seqfold import dg, fold
from utils import *

try:
    from nupack import *
except:
    print(
        "COULD NOT IMPORT NUPACK ON THIS DEVICE - proceeding, but will crash with nupack oracle selected"
    )
    pass
try:
    from bbdob import DeceptiveTrap, FourPeaks, NKLandscape, OneMax, TwoMin, WModel
    from bbdob.utils import idx2one_hot
except:
    print(
        "COULD NOT IMPORT BB-DOB ON THIS DEVICE - proceeding, but will crash with BB-DOB oracle selected"
    )
    pass


class Oracle:
    def __init__(
        self,
        oracle,
        seed=0,
        seq_len=30,
        dict_size=4,
        min_len=30,
        max_len=30,
        variable_len=True,
        init_len=0,
        energy_weight=False,
        nupack_target_motif="",
        seed_toy=0,
    ):
        """
        initialize the oracle
        :param config:
        """
        self.seed = seed
        self.seq_len = seq_len
        self.dict_size = dict_size
        self.min_len = min_len
        self.max_len = max_len
        self.init_len = init_len
        self.variable_len = variable_len
        self.oracle = oracle
        self.energy_weight = energy_weight
        self.nupack_target_motif = nupack_target_motif
        self.seed_toy = seed_toy

        np.random.seed(self.seed_toy)
        if not "nupack" in self.oracle:
            self.initRands() 

    def initRands(self):
        if True:
            self.linFactors = -np.ones(
                self.seq_len
            )
        else:
            self.linFactors = np.abs(
                np.random.randn(self.seq_len)
            )

        hamiltonian = np.random.randn(self.seq_len, self.seq_len)
        self.hamiltonian = (
            np.tril(hamiltonian) + np.tril(hamiltonian, -1).T
        )

        pham = np.zeros((self.seq_len, self.seq_len, self.dict_size, self.dict_size))
        for i in range(pham.shape[0]):
            for j in range(i, pham.shape[1]):
                for k in range(pham.shape[2]):
                    for l in range(k, pham.shape[3]):
                        num = -np.random.uniform(0, 1)
                        pham[i, j, k, l] = num
                        pham[i, j, l, k] = num
                        pham[j, i, k, l] = num
                        pham[j, i, l, k] = num
        self.pottsJ = (
            pham
        )
        self.pottsH = np.random.randn(
            self.seq_len, self.dict_size
        )

        # W-model parameters
        # first get the binary dimension size
        aa = np.arange(self.dict_size)
        if self.variable_len:
            aa = np.clip(aa, 1, self.dict_size)
        x0 = np.binary_repr(aa[-1])
        dimension = int(len(x0) * self.max_len)

        mu = np.random.randint(1, dimension + 1)
        v = np.random.randint(1, dimension + 1)
        m = np.random.randint(1, dimension)
        n = np.random.randint(1, dimension)
        gamma = np.random.randint(0, int(n * (n - 1) / 2))
        self.mu, self.v, self.m, self.n, self.gamma = [mu, v, m, n, gamma]

    def initializeDataset(
        self, save=True, returnData=False, customSize=None, custom_seed=None
    ):
        data = {}
        if custom_seed:
            np.random.seed(custom_seed)
        else:
            np.random.seed(self.seed)
        if customSize is None:
            datasetLength = self.init_len
        else:
            datasetLength = customSize

        if self.variable_len:
            samples = []
            while len(samples) < datasetLength:
                for i in range(self.min_len, self.max_len + 1):
                    samples.extend(
                        np.random.randint(
                            0 + 1,
                            self.dict_size + 1,
                            size=(int(10 * self.dict_size * i), i),
                        )
                    )

                samples = self.numpy_fillna(
                    np.asarray(samples, dtype=object)
                )
                samples = filterDuplicateSamples(
                    samples
                )
                if len(samples) < datasetLength:
                    samples = samples.tolist()
            np.random.shuffle(
                samples
            )
            samples = samples[
                :datasetLength
            ]
        else:
            samples = np.random.randint(
                1, self.dict_size + 1, size=(datasetLength, self.max_len)
            )
            samples = filterDuplicateSamples(samples)
            while len(samples) < datasetLength:
                samples = np.concatenate(
                    (
                        samples,
                        np.random.randint(
                            1, self.dict_size + 1, size=(datasetLength, self.max_len)
                        ),
                    ),
                    0,
                )
                samples = filterDuplicateSamples(samples)

        data["samples"] = samples
        data["energies"] = self.score(data["samples"])

        if save:
            np.save("datasets/" + self.oracle, data)
        if returnData:
            return data

    def score(self, queries):
        if isinstance(queries, list):
            queries = np.asarray(queries)
        block_size = int(1e4)
        scores_list = []
        scores_dict = {}
        for idx in range(len(queries) // block_size + bool(len(queries) % block_size)):
            queryBlock = queries[idx * block_size : (idx + 1) * block_size]
            scores_block = self.getScore(queryBlock)
            if isinstance(scores_block, dict):
                for k, v in scores_block.items():
                    if k in scores_dict:
                        scores_dict[k].extend(list(v))
                    else:
                        scores_dict.update({k: list(v)})
            else:
                scores_list.extend(self.getScore(queryBlock))
        if len(scores_list) > 0:
            return np.asarray(scores_list)
        else:
            return {k: np.asarray(v) for k, v in scores_dict.items()}

    def getScore(self, queries):
        if self.oracle == "linear":
            return self.linearToy(queries)
        elif self.oracle == "potts":
            return self.PottsEnergy(queries)
        elif self.oracle == "potts new":
            return self.PottsEnergyNew(queries)
        elif self.oracle == "inner product":
            return self.toyHamiltonian(queries)
        elif self.oracle == "seqfold":
            return self.seqfoldScore(queries)
        elif self.oracle == "nupack energy":
            return self.nupackScore(queries, returnFunc="energy")
        elif self.oracle == "nupack pins":
            return self.nupackScore(
                queries, returnFunc="pins", energy_weighting=self.energy_weight
            )
        elif self.oracle == "nupack pairs":
            return self.nupackScore(
                queries, returnFunc="pairs", energy_weighting=self.energy_weight
            )
        elif self.oracle == "nupack open loop":
            return self.nupackScore(
                queries, returnFunc="open loop", energy_weighting=self.energy_weight
            )
        elif self.oracle == "nupack motif":
            return self.nupackScore(
                queries,
                returnFunc="motif",
                motif=nupack_target_motif,
                energy_weighting=self.energy_weight,
            )

        elif (
            (self.oracle == "onemax")
            or (self.oracle == "twomin")
            or (self.oracle == "fourpeaks")
            or (self.oracle == "deceptivetrap")
            or (self.oracle == "nklandscape")
            or (self.oracle == "wmodel")
        ):
            return self.BB_DOB_functions(queries)
        elif isinstance(self.oracle, (list, ListConfig)) and all(
            ["nupack " in el for el in self.oracle]
        ):
            return self.nupackScore(
                queries, returnFunc=[el.replace("nupack ", "") for el in self.oracle]
            )
        elif (
            isinstance(self.oracle, (list, ListConfig))
            and self.oracle[0] == "potts new"
        ):
            return self.PottsEnergyNew(queries)
        else:
            raise NotImplementedError("Unknown oracle type")

    def BB_DOB_functions(self, queries):
        """
        BB-DOB OneMax benchmark
        :param queries:
        :return:
        """
        if self.variable_len:
            queries = np.clip(queries, 1, self.dict_size)  #  merge padding with class 1

        x0 = [
            np.binary_repr((queries[i][j] - 1).astype("uint8"), width=2)
            for i in range(len(queries))
            for j in range(self.max_len)
        ]
        x0 = (
            np.asarray(x0).astype(str).reshape(len(queries), self.max_len)
        )
        x0 = ["".join(x0[i]) for i in range(len(x0))]
        x1 = np.zeros((len(queries), len(x0[0])), int)
        for i in range(len(x0)):
            x1[i] = np.asarray(list(x0[i])).astype(int)

        dimension = x1.shape[1]

        x1 = idx2one_hot(x1, 2)  # convert to BB_DOB one_hot format

        objective = self.getObjective(dimension)

        evals, info = objective(x1)

        return evals

    def getObjective(self, dimension):
        if self.oracle == "onemax": 
            objective = OneMax(dimension)
        elif self.oracle == "twomin":
            objective = TwoMin(dimension)
        elif self.oracle == "fourpeaks":
            objective = FourPeaks(dimension, t=3)
        elif self.oracle == "deceptivetrap":
            objective = DeceptiveTrap(dimension, minimize=True)
        elif self.oracle == "nklandscape":
            objective = NKLandscape(dimension, minimize=True)
        elif self.oracle == "wmodel":
            objective = WModel(
                dimension,
                mu=self.mu,
                v=self.v,
                m=self.m,
                n=self.n,
                gamma=self.gamma,
                minimize=True,
            )
        else:
            printRecord(self.oracle + " is not a valid dataset!")
            sys.exit()

        return objective

    def linearToy(self, queries):
        energies = (
            queries @ self.linFactors
        )  # simple matmul - padding entries (zeros) have zero contribution

        return energies

    def toyHamiltonian(self, queries):
        energies = np.zeros(len(queries))
        for i in range(len(queries)):
            energies[i] = (
                queries[i] @ self.hamiltonian @ queries[i].transpose()
            )  # compute energy for each sample via inner product with the Hamiltonian

        return energies

    def PottsEnergy(self, queries):
        energies = np.zeros(len(queries))
        for k in range(len(queries)):
            nnz = np.count_nonzero(queries[k])
            for ii in range(nnz):
                energies[k] += self.pottsH[
                    ii, queries[k, ii] - 1
                ] 

                for jj in range(
                    ii, nnz
                ): 
                    energies[k] += (
                        2 * self.pottsJ[ii, jj, queries[k, ii] - 1, queries[k, jj] - 1]
                    )

        return energies

    def PottsEnergyNew(self, sequences):
        J, h = load_potts_model(435)

        energies = np.zeros(len(sequences))
        for idx, seq in enumerate(sequences):
            energies[idx] = potts_energy(J, h, seq)

        return energies

    def seqfoldScore(self, queries, returnSS=False):
        temperature = 37.0
        sequences = self.numbers2letters(queries)

        energies = np.zeros(len(sequences))
        strings = []
        pairLists = []
        i = -1
        for sequence in sequences:
            i += 1
            en = dg(
                sequence, temp=temperature
            ) 
            if np.isfinite(en):
                if (
                    en > 1500
                ): 
                    energies[i] = en - 1600
                else:
                    energies[i] = en
            else:
                energies[i] = 5

            if returnSS:
                structs = fold(sequence)

                desc = ["."] * len(sequence)
                pairList = []
                for s in structs:
                    pairList.append(s.ij[0])
                    if len(s.ij) == 1:
                        i, j = s.ij[0]
                        desc[i] = "("
                        desc[j] = ")"

                ssString = "".join(desc)
                strings.append(ssString)
                pairList = np.asarray(pairList) + 1
                pairLists.append(pairList)

        if returnSS:
            return energies, strings, pairLists
        else:
            return energies

    def numbers2letters(
        self, sequences
    ): 
        if type(sequences) != np.ndarray:
            sequences = np.asarray(sequences)

        my_seq = ["" for x in range(len(sequences))]
        row = 0
        for j in range(len(sequences)):
            seq = sequences[j, :]
            assert (
                type(seq) != str
            ), "Function inputs must be a list of equal length strings"
            for i in range(len(sequences[0])):
                na = seq[i]
                if na == 1:
                    my_seq[row] += "A"
                elif na == 2:
                    my_seq[row] += "T"
                elif na == 3:
                    my_seq[row] += "C"
                elif na == 4:
                    my_seq[row] += "G"
            row += 1
        return my_seq

    def numpy_fillna(self, data):
        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        out = np.zeros(mask.shape, dtype=object)
        out[mask] = np.concatenate(data)
        return out

    def nupackScore(
        self, queries, returnFunc="energy", energy_weighting=False, motif=None
    ):

        temperature = 310.0
        ionicStrength = 1.0
        if not isinstance(queries[0], str):
            sequences = self.numbers2letters(queries)
        else:
            sequences = queries

        energies = np.zeros(len(sequences))
        nPins = np.zeros(len(sequences)).astype(int)
        nPairs = 0
        ssStrings = np.zeros(len(sequences), dtype=object)

        # parallel evaluation - fast
        strandList = []
        comps = []
        i = -1
        for sequence in sequences:
            i += 1
            strandList.append(Strand(sequence, name="strand{}".format(i)))
            comps.append(Complex([strandList[-1]], name="comp{}".format(i)))

        set = ComplexSet(
            strands=strandList, complexes=SetSpec(max_size=1, include=comps)
        )
        model1 = Model(material="dna", celsius=temperature - 273, sodium=ionicStrength)
        results = complex_analysis(set, model=model1, compute=["mfe"])
        for i in range(len(energies)):
            energies[i] = results[comps[i]].mfe[0].energy
            ssStrings[i] = str(results[comps[i]].mfe[0].structure)

        dict_return = {}
        if "pins" in returnFunc:
            for i in range(len(ssStrings)):
                indA = 0  # hairpin completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == "(":
                        indA += 1
                    elif ssStrings[i][j] == ")":
                        indA -= 1
                        if indA == 0:  # if we come to the end of a distinct hairpin
                            nPins[i] += 1
            dict_return.update({"pins": -nPins})
        if "pairs" in returnFunc:
            nPairs = np.asarray([ssString.count("(") for ssString in ssStrings]).astype(
                int
            )
            dict_return.update({"pairs": -nPairs})
        if "energy" in returnFunc:
            dict_return.update(
                {"energy": energies}
            )

        if "open loop" in returnFunc:
            biggest_loop = np.zeros(len(ssStrings))
            for i in range(
                len(ssStrings)
            ): 
                loops = [0] 
                counting = 0
                indA = 0
                # loop completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == "(":
                        counting = 1
                        indA = 0
                    if (ssStrings[i][j] == ".") and (counting == 1):
                        indA += 1
                    if (ssStrings[i][j] == ")") and (counting == 1):
                        loops.append(indA)
                        counting = 0
                biggest_loop[i] = max(loops)
            dict_return.update({"open loop": -biggest_loop})

        if (
            "motif" in returnFunc
        ):
            padded_strings = bracket_dot_to_num(ssStrings, maxlen=self.max_len)
            padded_motif = np.expand_dims(
                bracket_dot_to_num([motif, motif], maxlen=self.max_len)[0], 0
            )
            motif_distance = binaryDistance(
                np.concatenate((padded_motif, padded_strings), axis=0), pairwise=True
            )[
                0, 1:
            ]
            dict_return.update(
                {"motif": motif_distance - 1}
            )
        if energy_weighting:
            for key in dict_return.keys():
                if key != "energy":
                    dict_return[key] = dict_return[key] * np.tanh(
                        np.abs(energies) / 2
                    )

        if isinstance(returnFunc, list):
            if len(returnFunc) > 1:
                return dict_return
            else:
                return dict_return[returnFunc[0]]
        else:
            return dict_return[returnFunc]
