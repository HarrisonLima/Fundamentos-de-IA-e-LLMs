import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");

// Pesos definidos para dar mais ou menos importância a certas características
// na hora de calcular a similaridade/recomendação.
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1,
};

// Normaliza valores contínuos (preço, idade) para o intervalo de 0 a 1.
// Por que? Mantém todas as features equilibradas para que valores muito altos
// (ex: um preço de 5000) não dominem o treinamento em relação a valores baixos (ex: idade 25).
// Fórmula: (valor - mínimo) / (máximo - mínimo)
const normalize = (value, min, max) => (value - min) / (max - min || 1);

// Analisa todos os produtos e usuários para extrair limites (mínimos e máximos)
// e criar índices numéricos para categorias e cores.
function makeContext(products, users) {
  // Extrai arrays apenas com idades e preços para encontrar os limites
  const ages = users.map((u) => u.age);
  const prices = products.map((p) => p.price);

  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);

  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  // Extrai valores únicos de cores e categorias usando Set
  const colors = [...new Set(products.map((p) => p.color))];
  const categories = [...new Set(products.map((p) => p.category))];

  // Cria dicionários de cor/categoria para índice numérico (ex: { "Vermelho": 0, "Azul": 1 })
  const colorsIndex = Object.fromEntries(
    colors.map((color, index) => {
      return [color, index];
    }),
  );

  const categoriesIndex = Object.fromEntries(
    categories.map((category, index) => {
      return [category, index];
    }),
  );

  /// Computar a média de idade dos usuários que compraram cada produto
  // Isso ajuda a IA a entender qual faixa etária prefere determinado item
  const midAge = (minAge + maxAge) / 2;
  const ageSums = {};
  const ageCounts = {};

  // Conta quantas vezes cada produto foi comprado e soma as idades dos compradores
  users.forEach((user) => {
    user.purchases.forEach((p) => {
      ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
      ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
    });
  });

  // Calcula a média final de idade por produto e já a normaliza (0 a 1)
  const productAvgAgeNorm = Object.fromEntries(
    products.map((product) => {
      const avg = ageCounts[product.name]
        ? ageSums[product.name] / ageCounts[product.name]
        : midAge; // Se o produto nunca foi comprado, assume a média de idade geral
      return [product.name, normalize(avg, minAge, maxAge)];
    }),
  );

  return {
    products,
    users,
    colorsIndex,
    categoriesIndex,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.length,
    numColors: colors.length,
    productAvgAgeNorm,
    // Dimensões totais = 2 (preço e idade) + total de categorias + total de cores
    dimentions: 2 + categories.length + colors.length,
  };
}

// Transforma um índice em um vetor One-Hot (ex: [0, 1, 0, 0]) e aplica o peso daquela característica.
const oneHotWeighted = (index, length, weight) =>
  tf.oneHot(index, length).cast("float32").mul(weight);

// Converte um produto em um vetor matemático (Tensor 1D) legível para a IA.
function encodeProduct(product, context) {
  // Normaliza o preço de 0 a 1 e aplica o peso
  const price = tf.tensor1d([
    normalize(product.price, context.minPrice, context.maxPrice) *
      WEIGHTS.price,
  ]);

  const age = tf.tensor1d([
    (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age,
  ]);

  // Converte categoria e cor em vetores One-Hot com pesos
  const category = oneHotWeighted(
    context.categoriesIndex[product.category],
    context.numCategories,
    WEIGHTS.category,
  );

  const color = oneHotWeighted(
    context.colorsIndex[product.color],
    context.numColors,
    WEIGHTS.color,
  );

  // Junta todas as características em uma única linha reta de números
  return tf.concat1d([price, age, category, color]);
}

// Converte um usuário em um vetor matemático (Tensor 1D).
// O "perfil" do usuário é calculado tirando a média matemática de todos os produtos que ele já comprou.
function encodeUser(user, context) {
  if (user.purchases.length) {
    return tf
      .stack(user.purchases.map((product) => encodeProduct(product, context)))
      .mean(0) // Calcula a média ao longo das colunas, criando um vetor único
      .reshape([1, context.dimentions]); // Força o formato final para [1 linha, X colunas]
  }

  return tf
    .concat1d([
      tf.zeros([1]),
      tf.tensor1d([
        normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age,
      ]),
      tf.zeros([context.numCategories]),
      tf.zeros([context.numColors]),
    ])
    .reshape([1, context.dimentions]);
}

// Fabrica os dados de treinamento cruzando cada usuário com todos os produtos.
function createTrainingData(context) {
  const inputs = []; // O que a IA recebe (Vetor Usuário + Vetor Produto)
  const labels = []; // A resposta certa (1 = Comprou, 0 = Não comprou)

  context.users
    .filter((u) => u.purchases.length)
    .forEach((user) => {
      // dataSync() extrai os números do Tensor para uma array normal do JavaScript
      const useVector = encodeUser(user, context).dataSync();
      context.products.forEach((product) => {
        const productVector = encodeProduct(product, context).dataSync();

        // Verifica se o usuário comprou este produto específico
        const label = user.purchases.some((purchase) =>
          purchase.name === product.name ? 1 : 0,
        );

        inputs.push([...useVector, ...productVector]);
        labels.push(label);
      });
    });

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    // tamanho = userVector + productVector
    inputDimention: context.dimentions * 2,
  };
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,'
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (acessórios = ativo)
//   1, 0, 0          // one-hot de cores (preto e cinza ativos, azul inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================

// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainData) {
  const model = tf.sequential();

  // Camada de entrada
  // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
  //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
  // - units: 128 neurônios (muitos "olhos" para detectar padrões)
  // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
  model.add(
    tf.layers.dense({
      inputShape: [trainData.inputDimention],
      units: 128,
      activation: "relu",
    }),
  );

  // Camada oculta 1
  // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
  // - activation: 'relu' (ainda extraindo combinações relevantes de features)
  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    }),
  );

  // Camada oculta 2
  // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
  //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
  // - activation: 'relu'
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
    }),
  );

  // Camada de saída
  // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
  // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
  //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch: epoch,
          loss: logs.loss,
          accuracy: logs.acc,
        });
      },
    },
  });

  return model;
}

// Função principal acionada pelo Web Worker para preparar os dados e (futuramente) treinar o modelo.
async function trainModel({ users }) {
  console.log("Training model with users:", users);
  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 50 },
  });
  const products = await (await fetch("/data/products.json")).json();

  const context = makeContext(products, users);

  // Pré-calcula os vetores de todos os produtos para uso futuro na recomendação
  context.productVectors = products.map((product) => {
    return {
      name: product.name,
      meta: { ...product },
      vector: encodeProduct(product, context).dataSync(),
    };
  });

  _globalCtx = context;

  const trainData = createTrainingData(context);
  _model = await configureNeuralNetAndTrain(trainData);

  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 100 },
  });
  postMessage({ type: workerEvents.trainingComplete });
}

function recommend({ user }) {
  if (!_model) return;
  const context = _globalCtx;

  // Converta o usuário fornecido no vetor de features codificadas
  // (preço ignorado, idade normalizada, categorias e cores ignoradas)
  // Isso transforma as informações do usuário no mesmo formato numérico que foi usado para treinar o modelo
  const userVector = encodeUser(user, context).dataSync();

  // Em aplicações reais:
  // Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
  // Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
  // Execute _model.predict() apenas nesses produtos

  // Crie pares de entrada: para cada produto, concatene o vetor do usuário
  // com o vetor codificado do produto.
  // Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).
  const input = context.productVectors.map(({ vector }) => {
    return [...userVector, ...vector];
  });

  // Converta todos esses pares (usuário, produto) em um único Tensor.
  // Formato: [numProdutos, inputDim]
  const inputTensor = tf.tensor2d(input);

  // Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
  // O resultado é uma pontuação para cada produto entre 0 e 1.
  // Quanto maior, maior a probabilidade do usuário querer aquele produto.
  const predictions = _model.predict(inputTensor);

  // Extraia as pontuações para um array JS normal.
  const scores = predictions.dataSync();
  const recommendations = context.productVectors.map((item, index) => {
    return {
      ...item.meta,
      name: item.name,
      score: scores[index], // previsão do modelo para este produto
    };
  });

  const sortedItems = recommendations.sort((a, b) => b.score - a.score);

  // Envie a lista ordenada de produtos recomendados
  // para a thread principal (a UI pode exibi-los agora).
  postMessage({
    type: workerEvents.recommend,
    user,
    recommendations: sortedItems,
  });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: recommend,
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
