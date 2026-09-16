import React, { useState, useEffect } from 'react';
import './index.css';
import { Search, X, ShoppingBasket, QrCode, Link as LinkIcon } from 'lucide-react';

const BACKEND_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8000' : 'https://ratatouille-backend.onrender.com';

const INDIAN_STATES = [
  "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
  "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
  "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
  "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
  "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
  "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
];

const COMMON_INGREDIENTS = [
  "Salt", "Olive oil", "Butter", "Garlic", "Onion", "Black pepper", "Sugar", "Water", "Lemon juice", "Tomato", "Egg", "Flour", "Milk", "Vegetable oil", "Parmesan cheese", "Parsley", "Soy sauce", "Chicken breast", "Brown sugar", "Vanilla extract", "Cumin", "Mayonnaise", "Paprika", "Oregano", "Cilantro", "Cinnamon", "Cheddar cheese", "Carrot", "Heavy cream", "Red pepper flakes", "Bell pepper", "Lemon", "Lime", "Chili powder", "Baking powder", "Basil", "Scallions", "Garlic powder", "Honey", "Balsamic vinegar", "Ginger", "Mozzarella cheese", "Red onion", "Baking soda", "Bacon", "White wine", "Chicken broth", "Thyme", "Cornstarch", "Sour cream", "Mustard", "Vinegar", "Celery", "Cayenne pepper", "Ground beef", "Nutmeg", "Coriander", "Almonds", "Apple cider vinegar", "Avocado", "Sesame oil", "Rosemary", "Soy", "Green onion", "Red wine vinegar", "Peanut butter", "Jalapeno", "Maple syrup", "Tomato paste", "Bread crumbs", "Lime juice", "Shrimp", "Dijon mustard", "Spinach", "Bay leaf", "White sugar", "Mint", "Sriracha", "Mushroom", "Walnut", "Chocolate chips", "Cream cheese", "Zucchini", "Beef broth", "Canola oil", "Coconut oil", "Potato", "Peanuts", "Cocoa powder", "Cucumber", "Clove", "Orange juice", "White pepper", "Apple", "Cabbage", "Oats", "Coconut milk", "Salmon", "Feta cheese", "Pork", "Blueberries", "Corn", "Sesame seeds", "Prosciutto", "Ketchup", "Yogurt", "Mustard powder", "Vegetable broth", "Strawberries", "Asparagus", "Pineapple", "Cauliflower", "Fish sauce", "Broccoli", "Allspice", "Avocado oil", "Black beans", "Almond milk", "Green beans", "Turmeric", "Chicken", "Ghee", "Bread", "Red wine", "Rice", "Pasta", "Chickpeas", "Beef", "Orange", "Sausage", "Maple", "Lime zest", "Lemon zest", "Dill", "Fennel", "Cardamom", "Anise", "Tarragon", "Chives", "Marjoram", "Sage", "Lemongrass", "Saffron", "Star anise", "Curry powder", "Garam masala", "Five spice", "Chili flakes", "Chipotle", "Habanero", "Poblano", "Serrano", "Gochugaru", "Anchovy", "Capers", "Olives", "Sun-dried tomatoes", "Artichoke", "Eggplant", "Butternut squash", "Sweet potato", "Pumpkin", "Radish", "Turnip", "Parsnip", "Beetroot", "Leek", "Shallot", "Watercress", "Arugula", "Kale", "Romaine", "Iceberg", "Radicchio", "Endive", "Brussels sprouts", "Bok choy", "Swiss chard", "Collard greens", "Mustard greens", "Cabbage (red)", "Cabbage (Napa)", "Seaweed", "Tofu", "Tempeh", "Seitan", "Edamame", "Lentils (red)", "Lentils (green)", "Lentils (brown)", "Beans (kidney)", "Beans (pinto)", "Beans (navy)", "Beans (cannellini)", "Peas", "Snow peas", "Snap peas", "Corn (sweet)", "Rice (white)", "Rice (brown)"
];

const UNSPLASH_CACHE = {
    "Curry": ["1585937421612-70a008356fbe", "1603894584373-5ac82b2ae398", "1565557623262-b51c2513a641", "1618160702438-9b02ab6515c9", "1631452180519-c014fe946bc7"],
    "Salad": ["1512621776951-a57141f2eefd", "1490645935967-10de6ba17061", "1546069901-ba9599a7e63c"],
    "Dessert": ["1488477181946-6428a0291777", "1550617931-e17a7b70dce2", "1551024506-0cb4a1cb3689"],
    "Bread": ["1509440159596-0249088772ff", "1608198093002-ad4e005484ec"],
    "Soup": ["1547592166-23ac45744acd", "1604152135912-04a022e23696", "1578020190125-f4f7c18bc9cb"],
    "Rice_Dish": ["1512058564366-18510be2db19", "1596797038530-2c107229654b"],
    "Dry_Sabzi": ["1546833999-b9f581a1996d", "1565557623262-b51c2513a641", "1604152135912-04a022e23696"]
};

const getRandomBanner = (archetype, isVegan = false) => {
    let arch = UNSPLASH_CACHE[archetype] ? archetype : "Curry";
    if (isVegan && arch === "Rice_Dish") {
        arch = "Dry_Sabzi";
    }
    const photos = UNSPLASH_CACHE[arch];
    const photoId = photos[Math.floor(Math.random() * photos.length)];
    return `https://images.unsplash.com/photo-${photoId}?q=80&w=1632&auto=format&fit=crop`;
};

function MultiSelectIngredient({ selected, setSelected }) {
  const [inputValue, setInputValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = React.useRef(null);

  const filtered = COMMON_INGREDIENTS.filter(
    ing => ing.toLowerCase().includes(inputValue.toLowerCase()) && !selected.includes(ing)
  ).slice(0, 8); // top 8 suggestions

  const addIngredient = (ing) => {
    const trimmed = ing.trim();
    if (trimmed && !selected.includes(trimmed)) {
      setSelected([...selected, trimmed]);
    }
    setInputValue('');
    // Keep suggestions open so user can pick another ingredient immediately
    inputRef.current?.focus();
  };

  const removeIngredient = (ing) => {
    setSelected(selected.filter(i => i !== ing));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addIngredient(inputValue);
    } else if (e.key === 'Backspace' && inputValue === '' && selected.length > 0) {
      removeIngredient(selected[selected.length - 1]);
    }
  };

  return (
    <div className="multi-select-container" style={{position: 'relative', width: '100%'}}>
      <div className="multi-select-input-wrapper" style={{
        display: 'flex', flexWrap: 'wrap', gap: '0.5rem', padding: '0.75rem 1rem',
        border: '1px solid #e5e7eb', borderRadius: '8px', background: '#fff', minHeight: '52px', alignItems: 'center'
      }}>
        {selected.map((ing, idx) => (
          <span key={idx} style={{
            background: '#fee2e2', color: '#b91c1c', padding: '0.2rem 0.6rem',
            borderRadius: '16px', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.3rem'
          }}>
            {ing}
            <span style={{cursor: 'pointer', opacity: 0.6}} onClick={() => removeIngredient(ing)}>×</span>
          </span>
        ))}
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
            setShowSuggestions(true);
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setShowSuggestions(false)}
          placeholder={selected.length === 0 ? "Type an ingredient and press Enter..." : ""}
          style={{
            flex: 1, border: 'none', outline: 'none', minWidth: '150px', fontSize: '1rem', background: 'transparent'
          }}
        />
      </div>
      {showSuggestions && (inputValue || filtered.length > 0) && (
        <ul style={{
          position: 'absolute', top: '100%', left: 0, right: 0, background: '#fff',
          border: '1px solid #e5e7eb', borderRadius: '8px', marginTop: '0.25rem',
          maxHeight: '200px', overflowY: 'auto', zIndex: 50, listStyle: 'none', padding: '0.5rem 0',
          boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'
        }}>
          {filtered.map((sug, idx) => (
            <li key={idx} onMouseDown={(e) => { e.preventDefault(); addIngredient(sug); }} style={{
              padding: '0.5rem 1rem', cursor: 'pointer', transition: 'background 0.2s'
            }} onMouseOver={(e) => e.target.style.background = '#f3f4f6'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
              {sug}
            </li>
          ))}
          {inputValue && !filtered.includes(inputValue) && (
             <li onMouseDown={(e) => { e.preventDefault(); addIngredient(inputValue); }} style={{
              padding: '0.5rem 1rem', cursor: 'pointer', color: '#ef4444', fontStyle: 'italic'
            }} onMouseOver={(e) => e.target.style.background = '#f3f4f6'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
              Add custom "{inputValue}"
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

function App() {
  const [viewMode, setViewMode] = useState('generate'); // 'generate' | 'vegan' | 'history'
  const [username, setUsername] = useState('');
  
  // Form State
  const [ingredients, setIngredients] = useState(['Rice', 'Egg', 'Potato', 'Tomato', 'Onion']);
  const [budget, setBudget] = useState(150);
  const [servings, setServings] = useState(1);
  const [stateName, setStateName] = useState('Delhi');
  const [modelVersion, setModelVersion] = useState('v10');  // 'v8' | 'v10'
  const [isVegan, setIsVegan] = useState(false);
  
  // Response State
  const [loading, setLoading] = useState(false);
  const [stepMessage, setStepMessage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [saveMessage, setSaveMessage] = useState('');
  
  // History State
  const [history, setHistory] = useState([]);

  // Vegan Engine State
  const [veganIngredients, setVeganIngredients] = useState('paneer, chicken');
  const [veganArchetype, setVeganArchetype] = useState('Curry');
  const [veganResult, setVeganResult] = useState(null);
  const [veganLoading, setVeganLoading] = useState(false);
  const [veganError, setVeganError] = useState('');

  // Community State
  const [globalRecipes, setGlobalRecipes] = useState([]);
  const [globalLoading, setGlobalLoading] = useState(false);

  // Vegan DB State (Kept for reference but not rendered)
  const [dbIngredient, setDbIngredient] = useState('');
  const [dbAlternatives, setDbAlternatives] = useState(null);
  const [dbAltLoading, setDbAltLoading] = useState(false);
  const [dbAltError, setDbAltError] = useState('');
  const [dbAltSource, setDbAltSource] = useState('');
  const [indianRecipes, setIndianRecipes] = useState([]);
  const [indianStyle, setIndianStyle] = useState('');
  const [indianLoading, setIndianLoading] = useState(false);
  const [indianTotal, setIndianTotal] = useState(0);
  const [expandedRecipe, setExpandedRecipe] = useState(null);

  // Timer State
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    let interval = null;
    if (loading || veganLoading || dbAltLoading || indianLoading) {
      interval = setInterval(() => {
        setElapsedSeconds(prev => prev + 1);
      }, 1000);
    } else {
      setElapsedSeconds(0);
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [loading, veganLoading, dbAltLoading, indianLoading]);

  // Pre-fetch community recipes on mount for the Surprise Me features
  useEffect(() => {
    const loadCommunityData = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/all-recipes`);
        const data = await response.json();
        if (data.status === 'success') {
          setGlobalRecipes(data.recipes);
        }
      } catch (err) {
        console.error("Failed to load global recipes", err);
      }
    };
    loadCommunityData();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    setSaveMessage('');
    setStepMessage('Initializing...');

    const ingList = Array.isArray(ingredients) ? ingredients : ingredients.split(',').map(i => i.trim()).filter(i => i);

    try {
      const response = await fetch(`${BACKEND_URL}/generate-recipe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ingredients: ingList,
          budget: parseFloat(budget),
          servings: parseInt(servings),
          state: stateName,
          model_version: modelVersion,
          is_vegan: isVegan
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        
        const lines = chunk.split('\n\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (!dataStr.trim()) continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.step === 'complete') {
                setResult(data.result);
                setStepMessage('');
              } else if (data.step === 'error') {
                setError(data.message);
                setStepMessage('');
              } else {
                setStepMessage(data.message);
              }
            } catch (e) {
              console.warn("Failed to parse chunk:", dataStr);
            }
          }
        }
      }

    } catch (err) {
      setError('Connection to backend failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveRecipe = async () => {
    if (!username) {
      alert("Please enter a username to save this recipe!");
      return;
    }
    
    try {
      const response = await fetch(`${BACKEND_URL}/save-recipe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: username,
          recipe_data: result
        })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setSaveMessage('✅ Recipe saved successfully!');
      } else {
        alert(data.message);
      }
    } catch (err) {
      alert("Failed to connect to database.");
    }
  };

  const fetchHistory = async () => {
    if (!username) {
      alert("Please enter a username first!");
      return;
    }
    
    setLoading(true);
    setViewMode('history');
    
    try {
      const response = await fetch(`${BACKEND_URL}/my-recipes/${username}`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setHistory(data.recipes);
      } else {
        alert(data.message);
      }
    } catch (err) {
      alert("Failed to fetch history.");
    } finally {
      setLoading(false);
    }
  };

  const fetchGlobalRecipes = async () => {
    setGlobalLoading(true);
    setViewMode('community');
    
    try {
      const response = await fetch(`${BACKEND_URL}/all-recipes`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setGlobalRecipes(data.recipes);
      } else {
        alert(data.message);
      }
    } catch (err) {
      alert("Failed to fetch community recipes.");
    } finally {
      setGlobalLoading(false);
    }
  };

  const handleVeganConvert = async (e) => {
    e.preventDefault();
    setVeganLoading(true);
    setVeganError('');
    setVeganResult(null);

    const list = veganIngredients.split(',').map(i => i.trim()).filter(i => i);

    try {
      const response = await fetch(`${BACKEND_URL}/get-vegan-blueprint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ingredients: list,
          archetype: veganArchetype
        })
      });

      const data = await response.json();

      if (data.status === 'success') {
        setVeganResult(data.results);
      } else {
        setVeganError(data.message || 'Failed to generate vegan blueprint.');
      }
    } catch (err) {
      setVeganError('Connection to backend failed. Please make sure local api server is running.');
    } finally {
      setVeganLoading(false);
    }
  };

  const handleSurpriseMe = () => {
    if (globalRecipes.length === 0) {
      alert("Community recipes are still loading. Please try again in a moment.");
      return;
    }
    const randomDoc = globalRecipes[Math.floor(Math.random() * globalRecipes.length)];
    setResult({
      ...randomDoc.recipe,
      isSurprise: true
    });
  };

  return (
    <div className="app-container">
      <div className="hero-bg">
        <nav className="top-nav">
          <div className="nav-left">
            <div className="logo-section" onClick={() => setViewMode('generate')}>
              <img src="/lab-logo.png" alt="Cosyslab Logo" style={{ width: '32px', height: '32px' }} />
              Ratatouille
            </div>
            
            <div className="nav-links" style={{ marginLeft: '2rem' }}>
              <button 
                className={`nav-tab ${viewMode === 'generate' ? 'active' : ''}`}
                onClick={() => setViewMode('generate')}
              >
                Cook
              </button>
              <button 
                className={`nav-tab ${viewMode === 'history' ? 'active' : ''}`}
                onClick={() => {
                  setViewMode('history');
                  fetchHistory();
                }}
              >
                My Recipes
              </button>
              <button 
                className={`nav-tab ${viewMode === 'community' ? 'active' : ''}`}
                onClick={() => fetchGlobalRecipes()}
              >
                Community
              </button>
            </div>
          </div>

          <div className="nav-right">
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <a href="https://twitter.com/intent/tweet?url=https%3A%2F%2Fcosylab.iiitd.edu.in%2Fratatouille&text=AI-Generated%20Recipes%20using%20%23Ratatouille.%0A%0AComputational%20Gastronomy%2C%20a%20data%20science%20of%20food%2C%20flavors%2C%20nutritions%2C%20health%2C%20and%20sustainability.%0A%0AMaking%20Food%20Computable.%20Complex%20Systems%20Laboratory%2C%20IIIIT-Delhi%0A%0A" target="_blank" rel="noopener noreferrer" className="social-icon" style={{ display: 'flex' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="#1DA1F2"><path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/></svg>
              </a>
              <a href="https://www.facebook.com/sharer/sharer.php?u=https://cosylab.iiitd.edu.in/ratatouille&t=AI-Generated%20Recipes%20using%20%23Ratatouille.%0A%0AComputational%20Gastronomy%2C%20a%20data%20science%20of%20food%2C%20flavors%2C%20nutritions%2C%20health%2C%20and%20sustainability.%0A%0AMaking%20Food%20Computable.%20Complex%20Systems%20Laboratory%2C%20IIIIT-Delhi%0A%0A" target="_blank" rel="noopener noreferrer" className="social-icon" style={{ display: 'flex' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="#1877F2"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.469h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.469h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
              </a>
              <a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fcosylab.iiitd.edu.in%2Fratatouille" target="_blank" rel="noopener noreferrer" className="social-icon" style={{ display: 'flex' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="#0A66C2"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
              </a>
              <div onClick={() => navigator.clipboard.writeText('https://cosylab.iiitd.edu.in/ratatouille')} title="Copy Link">
                <LinkIcon className="social-icon" style={{ color: '#ef4444', cursor: 'pointer' }} size={22} />
              </div>
            </div>
          </div>
        </nav>

        {viewMode === 'generate' && (
          <div className="hero-content fade-in">
            <h1 className="title">Create Novel Recipes</h1>
            <p className="subtitle">
              Stuck in a dinner dilemma?<br />
              Generate novel recipes with ingredients of your choice.
            </p>
          </div>
        )}
      </div>

      {viewMode === 'generate' && (
        <div className="main-panel fade-in" style={{ marginTop: 0 }}>
          <div className="search-container">
            <div className="search-bar-wrapper">
              <MultiSelectIngredient selected={ingredients} setSelected={setIngredients} />
            </div>

            <div className="controls-row">
              <div className="control-item">
                <label>Budget (₹)</label>
                <input 
                  type="number" 
                  value={budget} 
                  onChange={(e) => setBudget(e.target.value)} 
                  min="10"
                />
              </div>
              <div className="control-item">
                <label>Servings</label>
                <input 
                  type="number" 
                  value={servings} 
                  onChange={(e) => setServings(e.target.value)} 
                  min="1"
                />
              </div>
              <div className="control-item">
                <label>Region</label>
                <select 
                  value={stateName} 
                  onChange={(e) => setStateName(e.target.value)}
                  style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '1rem', color: 'var(--text-main)', cursor: 'pointer' }}
                >
                  {INDIAN_STATES.map(state => (
                    <option key={state} value={state}>{state}</option>
                  ))}
                </select>
              </div>
              <label className="vegan-toggle">
                <input 
                  type="checkbox" 
                  checked={isVegan} 
                  onChange={(e) => setIsVegan(e.target.checked)} 
                />
                Make it Vegan
              </label>
              <div className="model-toggle">
                <button type="button" className={modelVersion === 'v8' ? 'active' : ''} onClick={() => setModelVersion('v8')}>V8</button>
                <button type="button" className={modelVersion === 'v10' ? 'active' : ''} onClick={() => setModelVersion('v10')}>V10</button>
              </div>
            </div>

            <div className="action-buttons">
              <button onClick={handleSubmit} disabled={loading} className="btn-primary">
                {loading ? <span className="loader"></span> : 'Generate Recipe'}
              </button>
              <button onClick={handleSurpriseMe} className="btn-secondary">
                Surprise me!!
              </button>
            </div>

            <div className="chips-outer-container" style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '2.5rem', maxWidth: '800px', width: '100%' }}>
              <div className="chip highlight" onClick={handleSurpriseMe} style={{ flexShrink: 0, zIndex: 10, boxShadow: '0 4px 6px -1px rgba(250, 204, 21, 0.4)' }}>✨ Random</div>
              <div className="marquee-wrapper" style={{ marginTop: 0, flex: 1 }}>
                <div className="chips-container marquee-content">
                  {globalRecipes.slice(0, 10).map((doc, idx) => {
                     const title = doc.recipe.recipe.split('\n')[0].replace('### TITLE:', '').trim();
                     return (
                       <div key={`a-${idx}`} className="chip" onClick={() => {
                         const imgUrl = doc.recipe.image_url || getRandomBanner(doc.recipe.archetype, doc.recipe.is_vegan);
                         setResult({...doc.recipe, image_url: imgUrl, isSurprise: true});
                       }} style={{ flexShrink: 0 }}>
                         {title}
                       </div>
                     );
                  })}
                  {/* Duplicated for infinite scrolling effect */}
                  {globalRecipes.slice(0, 10).map((doc, idx) => {
                     const title = doc.recipe.recipe.split('\n')[0].replace('### TITLE:', '').trim();
                     return (
                       <div key={`b-${idx}`} className="chip" onClick={() => {
                         const imgUrl = doc.recipe.image_url || getRandomBanner(doc.recipe.archetype, doc.recipe.is_vegan);
                         setResult({...doc.recipe, image_url: imgUrl, isSurprise: true});
                       }} style={{ flexShrink: 0 }}>
                         {title}
                       </div>
                     );
                  })}
                </div>
              </div>
            </div>
          </div>

          {error && <div style={{ color: 'red', marginTop: '1rem' }}>⚠️ {error}</div>}

          {result && (
            <div className="modal-overlay">
              <div className="modal-content fade-in">
                <div className="modal-header">
                  <h2 className="modal-title">{result.recipe.split('\n')[0].replace('### TITLE:', '').trim()}</h2>
                  <button className="close-btn" onClick={() => setResult(null)}><X size={24} /></button>
                </div>
                
                {(result.image_url || result.isSurprise || result.archetype) ? (
                  <div className="modal-image-container">
                    <img src={result.image_url || getRandomBanner(result.archetype, result.is_vegan)} alt="Recipe" className="modal-image" />
                  </div>
                ) : (
                  <div style={{ marginTop: '70px' }}></div>
                )}
                
                <div className="modal-body">
                  <div className="ingredients-col">
                    <h3>Ingredients</h3>
                    <ul className="ingredients-list">
                      {result.calculated_ingredients.map((ing, idx) => (
                        <li key={idx} className="ingredient-item">
                          <ShoppingBasket className="basket-icon" size={20} />
                          <span>{ing}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="instructions-col">
                    <h3>Instructions</h3>
                    <ul className="instructions-list">
                      {result.recipe.includes('### DIRECTIONS:') 
                        ? result.recipe.split('### DIRECTIONS:')[1].trim().split('\n').map((step, idx) => {
                            const match = step.match(/^(\d+)\.\s+(.*)/);
                            if (match) {
                              return <li key={idx}><strong>{match[1]}.</strong> <span>{match[2]}</span></li>;
                            }
                            return <li key={idx}><span>{step}</span></li>;
                          })
                        : <li><span>{result.recipe}</span></li>}
                    </ul>
                  </div>
                </div>

                <div className="modal-footer" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
                  <button className="btn-regenerate" onClick={() => { setResult(null); setSaveMessage(''); }}>Regenerate Recipe</button>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    {saveMessage ? (
                      <span style={{ color: '#10b981', fontWeight: 'bold' }}>{saveMessage}</span>
                    ) : (
                      <>
                        <input 
                          type="text" 
                          placeholder="Enter your username..." 
                          value={username} 
                          onChange={(e) => setUsername(e.target.value)} 
                          style={{ padding: '0.75rem', borderRadius: '8px', border: '1px solid #e5e7eb', outline: 'none', fontSize: '0.95rem' }}
                        />
                        <button 
                          style={{ background: '#3b82f6', color: 'white', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }} 
                          onClick={handleSaveRecipe}
                        >
                          Save Recipe
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {viewMode === 'vegan' && (
        <>
          <div className="glass-panel main-panel fade-in">
            <header>
              <h1 className="title">Vegan Substitution Engine 🌿</h1>
              <p className="subtitle">Find the best physical-chemical plant alternatives</p>
            </header>

            <form onSubmit={handleVeganConvert} className="recipe-form">
              <div className="input-group">
                <label>Ingredients to Convert (comma separated)</label>
                <input 
                  type="text" 
                  value={veganIngredients} 
                  onChange={(e) => setVeganIngredients(e.target.value)} 
                  required 
                  placeholder="e.g. paneer, chicken, butter, egg"
                />
              </div>

              <div className="input-group">
                <label>Dish Archetype (determines technique context)</label>
                <select 
                  value={veganArchetype} 
                  onChange={(e) => setVeganArchetype(e.target.value)}
                  className="username-input"
                  style={{ width: '100%', padding: '0.8rem', background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#fff', fontSize: '1rem', outline: 'none' }}
                >
                  <option value="Curry" style={{ background: '#1f2937' }}>Curry</option>
                  <option value="Dry_Sabzi" style={{ background: '#1f2937' }}>Dry Sabzi</option>
                  <option value="Salad" style={{ background: '#1f2937' }}>Salad</option>
                  <option value="Soup" style={{ background: '#1f2937' }}>Soup</option>
                  <option value="Dessert" style={{ background: '#1f2937' }}>Dessert</option>
                  <option value="Bread" style={{ background: '#1f2937' }}>Bread</option>
                  <option value="Rice_Dish" style={{ background: '#1f2937' }}>Rice Dish</option>
                </select>
              </div>

              <button type="submit" disabled={veganLoading} className="generate-btn">
                {veganLoading ? <span className="loader"></span> : 'Calculate Vegan Blueprint'}
              </button>
            </form>
          </div>

          {(veganResult || veganError) && (
            <div className="glass-panel result-panel fade-in">
              {veganError ? (
                <div className="error-message">⚠️ {veganError}</div>
              ) : (
                <div className="vegan-blueprint-results">
                  <h2 style={{ fontSize: '1.6rem', marginBottom: '1.5rem', textAlign: 'center', color: '#10b981' }}>
                    🌿 Calculated Substitution Blueprint
                  </h2>
                  {veganResult.map((res, idx) => (
                    <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', padding: '1.5rem', borderRadius: '12px', marginBottom: '1.5rem', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      {res.status === 'already_vegan' ? (
                        <div style={{ textAlign: 'center', color: '#10B981' }}>
                          <h3 style={{ margin: '0 0 0.5rem 0' }}>✓ {(res.original_ingredient || '').toUpperCase()}</h3>
                          <p style={{ margin: 0, opacity: 0.8 }}>This ingredient is already vegan!</p>
                        </div>
                      ) : res.status === 'error' ? (
                        <div style={{ color: '#EF4444' }}>
                          <h3 style={{ margin: '0 0 0.5rem 0' }}>⚠️ {(res.original_ingredient || '').toUpperCase()}</h3>
                          <p style={{ margin: 0, opacity: 0.8 }}>{res.message}</p>
                        </div>
                      ) : (
                        <div>
                          {/* Header */}
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.8rem', marginBottom: '1rem' }}>
                            <div>
                              <span style={{ fontSize: '0.8rem', opacity: 0.5 }}>Substitution:</span>
                              <h3 style={{ margin: 0, textTransform: 'capitalize', color: '#f3f4f6', fontSize: '1.2rem' }}>
                                {res.original_ingredient} → <span style={{ color: '#10b981' }}>{res.best_vegan_substitute}</span>
                              </h3>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                              <span style={{ fontSize: '0.8rem', opacity: 0.5 }}>Match Score:</span>
                              <h4 style={{ margin: 0, color: '#10b981', fontSize: '1.2rem' }}>{(res.match_score * 100).toFixed(0)}%</h4>
                            </div>
                          </div>

                          {/* Scores Breakdown */}
                          <div style={{ marginBottom: '1.2rem' }}>
                            <h4 style={{ margin: '0 0 0.6rem 0', fontSize: '0.9rem', opacity: 0.8 }}>Similarity Breakdown:</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                              <div>
                                <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>Aroma/Flavor Fit</span>
                                <div style={{ background: '#374151', height: '6px', borderRadius: '3px', marginTop: '4px' }}>
                                  <div style={{ width: `${res.score_breakdown.flavor_similarity * 100}%`, background: '#3b82f6', height: '100%', borderRadius: '3px' }}></div>
                                </div>
                                <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>{(res.score_breakdown.flavor_similarity * 100).toFixed(0)}%</span>
                              </div>
                              <div>
                                <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>Textural Similarity</span>
                                <div style={{ background: '#374151', height: '6px', borderRadius: '3px', marginTop: '4px' }}>
                                  <div style={{ width: `${res.score_breakdown.texture_similarity * 100}%`, background: '#8b5cf6', height: '100%', borderRadius: '3px' }}></div>
                                </div>
                                <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>{(res.score_breakdown.texture_similarity * 100).toFixed(0)}%</span>
                              </div>
                              <div>
                                <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>Culinary Role Match</span>
                                <div style={{ background: '#374151', height: '6px', borderRadius: '3px', marginTop: '4px' }}>
                                  <div style={{ width: `${res.score_breakdown.functional_fit * 100}%`, background: '#10b981', height: '100%', borderRadius: '3px' }}></div>
                                </div>
                                <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>{(res.score_breakdown.functional_fit * 100).toFixed(0)}%</span>
                              </div>
                            </div>
                          </div>

                          {/* Compensation Blueprint */}
                          <div style={{ background: 'rgba(0,0,0,0.15)', padding: '1rem', borderRadius: '8px' }}>
                            <h4 style={{ margin: '0 0 0.8rem 0', color: '#10b981', fontSize: '0.95rem' }}>Delta Compensation Steps:</h4>
                            
                            {/* Prep Techniques */}
                            {res.compensation_blueprint.techniques.length > 0 && (
                              <div style={{ marginBottom: '1rem' }}>
                                <h5 style={{ margin: '0 0 0.3rem 0', fontSize: '0.8rem', color: '#9ca3af' }}>Physical & Textural Prep:</h5>
                                <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.85rem', color: '#d1d5db' }}>
                                  {res.compensation_blueprint.techniques.map((tech, tIdx) => (
                                    <li key={tIdx} style={{ marginBottom: '0.3rem' }}>{tech}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Auxiliary Additions */}
                            {res.compensation_blueprint.auxiliary_additions.length > 0 && (
                              <div style={{ marginBottom: '1rem' }}>
                                <h5 style={{ margin: '0 0 0.3rem 0', fontSize: '0.8rem', color: '#9ca3af' }}>Recommended Compound Additions:</h5>
                                <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.85rem', color: '#d1d5db' }}>
                                  {res.compensation_blueprint.auxiliary_additions.map((add, aIdx) => (
                                    <li key={aIdx} style={{ marginBottom: '0.3rem' }}>
                                      <strong>{add.amount} {add.name}</strong> - <em>{add.purpose}</em>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Spice Bridges */}
                            {res.compensation_blueprint.spice_bridge.length > 0 && (
                              <div>
                                <h5 style={{ margin: '0 0 0.3rem 0', fontSize: '0.8rem', color: '#9ca3af' }}>Aromatic Spice Bridges:</h5>
                                <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.85rem', color: '#d1d5db' }}>
                                  {res.compensation_blueprint.spice_bridge.map((spice, sIdx) => (
                                    <li key={sIdx} style={{ marginBottom: '0.3rem' }}>
                                      Add <strong>{spice.spice}</strong> (covers {(spice.fills_gap_ratio * 100).toFixed(0)}% gap: <em>{spice.reason}</em>)
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}

      {viewMode === 'history' && (
        <div className="glass-panel main-panel fade-in">
          <header>
            <h1 className="title">My Recipes 📖</h1>
            <p className="subtitle">View saved recipes for your profile</p>
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginBottom: '2rem' }}>
              <input 
                type="text" 
                placeholder="Enter username..." 
                value={username} 
                onChange={(e) => setUsername(e.target.value)} 
                onKeyDown={(e) => e.key === 'Enter' && fetchHistory()}
                style={{ padding: '0.6rem 1rem', borderRadius: '20px', border: '1px solid #ccc', outline: 'none', width: '250px' }}
              />
              <button onClick={fetchHistory} style={{ padding: '0.6rem 1.5rem', borderRadius: '20px', background: '#7a1d41', color: 'white', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}>
                Load History
              </button>
            </div>
          </header>
          
          {loading ? (
             <div className="loader" style={{margin: '0 auto'}}></div>
          ) : history.length === 0 ? (
             <p style={{textAlign: 'center', color: '#ccc'}}>No recipes saved yet!</p>
          ) : (
            <div className="history-list">
              {history.map((doc, idx) => {
                 const rec = doc.recipe;
                 return (
                   <details key={idx} className="history-card">
                     <summary className="history-summary">
                       <div style={{display: 'flex', gap: '1.25rem', alignItems: 'center'}}>
                         <img src={rec.image_url || getRandomBanner(rec.archetype, rec.is_vegan)} alt="Thumbnail" style={{width: '70px', height: '70px', objectFit: 'cover', borderRadius: '12px'}} />
                         <div>
                           <h4>{rec.recipe.split('\n')[0].replace('### TITLE:', '').trim()}</h4>
                           <span className="badge">{rec.archetype}</span>
                           <p className="date-text">Saved: {new Date(doc.created_at * 1000).toLocaleString()}</p>
                         </div>
                       </div>
                       <div className="expand-hint">Click to view ▾</div>
                     </summary>
                     
                     <div className="history-expanded-content">
                        <div className="ingredients-box" style={{marginTop: '1rem'}}>
                          <h4>Ingredients</h4>
                          <ul>
                            {rec.calculated_ingredients.map((ing, i) => (
                              <li key={i}>✓ {ing}</li>
                            ))}
                          </ul>
                        </div>
                        
                        <div className="directions-box" style={{marginTop: '1rem'}}>
                          <h4>Directions</h4>
                          <div className="recipe-text" style={{fontSize: '0.95rem'}}>
                            {rec.recipe.includes('### DIRECTIONS:') 
                              ? rec.recipe.split('### DIRECTIONS:')[1].trim().split('\n').map((step, i) => (
                                  <p key={i} className="step-text" style={{padding: '0.8rem', marginBottom: '0.5rem'}}>{step}</p>
                                ))
                              : <p className="step-text">{rec.recipe}</p>}
                          </div>
                        </div>
                     </div>
                   </details>
                 )
              })}
            </div>
          )}
        </div>
      )}

      {viewMode === 'community' && (
        <div className="glass-panel main-panel fade-in">
          <header>
            <h1 className="title">Community Recipes 🌍</h1>
            <p className="subtitle">Explore all user-generated recipes</p>
          </header>
          
          {globalLoading ? (
             <div className="loader" style={{margin: '0 auto'}}></div>
          ) : globalRecipes.length === 0 ? (
             <p style={{textAlign: 'center', color: '#ccc'}}>No recipes found in the community.</p>
          ) : (
            <div className="history-list">
              {globalRecipes.map((doc, idx) => {
                 const rec = doc.recipe;
                 return (
                   <details key={idx} className="history-card" style={{border: rec.is_vegan ? '2px solid #4ade80' : 'none', position: 'relative'}}>
                     <summary className="history-summary">
                       <div style={{display: 'flex', gap: '1.25rem', alignItems: 'center'}}>
                         <img src={rec.image_url || getRandomBanner(rec.archetype, rec.is_vegan)} alt="Thumbnail" style={{width: '70px', height: '70px', objectFit: 'cover', borderRadius: '12px'}} />
                         <div>
                           <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                             <h4>{rec.recipe.split('\n')[0].replace('### TITLE:', '').trim()}</h4>
                             {rec.is_vegan && (
                               <span style={{background: '#4ade80', color: '#064e3b', padding: '2px 8px', borderRadius: '12px', fontSize: '0.8rem', fontWeight: 'bold'}}>
                                 🌱 VEGAN
                               </span>
                             )}
                           </div>
                           <span className="badge">{rec.archetype}</span>
                           <span style={{marginLeft: '10px', fontSize: '0.85rem', color: '#9ca3af'}}>
                             by <strong>{doc.username}</strong>
                           </span>
                           <p className="date-text">Saved: {new Date(doc.created_at * 1000).toLocaleString()}</p>
                         </div>
                       </div>
                       <div className="expand-hint">Click to view ▾</div>
                     </summary>
                     
                     <div className="history-expanded-content">
                        <div className="ingredients-box" style={{marginTop: '1rem'}}>
                          <h4>Ingredients</h4>
                          <ul>
                            {rec.calculated_ingredients.map((ing, i) => (
                              <li key={i}>✓ {ing}</li>
                            ))}
                          </ul>
                        </div>
                        
                        <div className="directions-box" style={{marginTop: '1rem'}}>
                          <h4>Directions</h4>
                          <div className="recipe-text" style={{fontSize: '0.95rem'}}>
                            {rec.recipe.includes('### DIRECTIONS:') 
                              ? rec.recipe.split('### DIRECTIONS:')[1].trim().split('\n').map((step, i) => (
                                  <p key={i} className="step-text" style={{padding: '0.8rem', marginBottom: '0.5rem'}}>{step}</p>
                                ))
                              : <p className="step-text">{rec.recipe}</p>}
                          </div>
                        </div>
                     </div>
                   </details>
                 )
              })}
            </div>
          )}
        </div>
      )}

      {(loading || globalLoading || veganLoading || dbAltLoading || indianLoading) && (
        <div className="floating-timer-tab">
          <span className="loader" style={{width:'16px',height:'16px',borderWidth:'2px',borderColor:'#fff',borderBottomColor:'transparent',display:'inline-block',boxSizing:'border-box',animation:'rotation 1s linear infinite',borderRadius:'50%'}}></span>
          <span className="timer-text">
            {loading ? (stepMessage || "Generating Recipe...") : 
             veganLoading ? "Calculating Vegan Match..." : 
             dbAltLoading || indianLoading || globalLoading ? "Loading Data..." : "Working..."}
            <span style={{marginLeft:'8px',color:'#818cf8',fontWeight:'600'}}>({elapsedSeconds}s)</span>
          </span>
        </div>
      )}

    </div>
  );
}

export default App;
