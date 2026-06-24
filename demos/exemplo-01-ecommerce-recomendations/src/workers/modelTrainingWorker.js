import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");

// Pesos definidos para dar mais ou menos importância a certas características
// na hora de calcular a similaridade/recomendação.
let _globalCtx = {};
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
}

// Fabrica os dados de treinamento cruzando cada usuário com todos os produtos.
function createTrainingData(context) {
  const inputs = []; // O que a IA recebe (Vetor Usuário + Vetor Produto)
  const labels = []; // A resposta certa (1 = Comprou, 0 = Não comprou)

  context.users.forEach((user) => {
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
    inputDimetion: context.dimentions * 2,
  };
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
  debugger;
  postMessage({
    type: workerEvents.trainingLog,
    epoch: 1,
    loss: 1,
    accuracy: 1,
  });

  setTimeout(() => {
    postMessage({
      type: workerEvents.progressUpdate,
      progress: { progress: 100 },
    });
    postMessage({ type: workerEvents.trainingComplete });
  }, 1000);
}
function recommend(user, ctx) {
  console.log("will recommend for user:", user);
  // postMessage({
  //     type: workerEvents.recommend,
  //     user,
  //     recommendations: []
  // });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data);
};
