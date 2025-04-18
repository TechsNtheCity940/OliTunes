// Collection of inspirational music quotes for UI display
const musicQuotes = [
  {
    quote: "Music gives a soul to the universe, wings to the mind, flight to the imagination and life to everything.",
    author: "Plato"
  },
  {
    quote: "One good thing about music, when it hits you, you feel no pain.",
    author: "Bob Marley"
  },
  {
    quote: "Music is a language that doesn't speak in particular words. It speaks in emotions.",
    author: "Keith Richards"
  },
  {
    quote: "Music is the universal language of mankind.",
    author: "Henry Wadsworth Longfellow"
  },
  {
    quote: "Without music, life would be a mistake.",
    author: "Friedrich Nietzsche"
  },
  {
    quote: "Music is the soundtrack of your life.",
    author: "Dick Clark"
  },
  {
    quote: "Music can change the world because it can change people.",
    author: "Bono"
  },
  {
    quote: "Where words fail, music speaks.",
    author: "Hans Christian Andersen"
  },
  {
    quote: "Music in the soul can be heard by the universe.",
    author: "Lao Tzu"
  },
  {
    quote: "If music be the food of love, play on.",
    author: "William Shakespeare"
  },
  {
    quote: "Life seems to go on without effort when I am filled with music.",
    author: "George Eliot"
  },
  {
    quote: "After silence, that which comes nearest to expressing the inexpressible is music.",
    author: "Aldous Huxley"
  },
  {
    quote: "Music is the divine way to tell beautiful, poetic things to the heart.",
    author: "Pablo Casals"
  },
  {
    quote: "Music, once admitted to the soul, becomes a sort of spirit, and never dies.",
    author: "Edward Bulwer-Lytton"
  },
  {
    quote: "Music is the wine that fills the cup of silence.",
    author: "Robert Fripp"
  },
];

// Helper functions for quotes
export const getRandomQuote = () => {
  return musicQuotes[Math.floor(Math.random() * musicQuotes.length)];
};

export const getQuoteByIndex = (index) => {
  if (index < 0 || index >= musicQuotes.length) {
    return getRandomQuote();
  }
  return musicQuotes[index];
};

export default musicQuotes;
