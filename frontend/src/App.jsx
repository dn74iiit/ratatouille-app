import React, { useState, useEffect } from 'react';
import './index.css';
import { Search, X, ShoppingBasket, QrCode, Link as LinkIcon, Share2, Globe, Check } from 'lucide-react';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8000' : 'https://ratatouille-backend.onrender.com');

const INDIAN_STATES = [
  "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
  "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
  "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
  "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
  "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
  "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
];

export const STATE_DEFAULT_INGREDIENTS = {
  "Andaman and Nicobar Islands": ["Rice", "Coconut oil", "Fish", "Tomato", "Curry leaves"],
  "Andhra Pradesh": ["Rice", "Toor Dal(Arhar)", "Tomato", "Chili powder", "Garlic", "Cumin seeds(Jeera)"],
  "Arunachal Pradesh": ["Rice", "Ginger", "Garlic", "Spinach", "Mustard oil", "Green Chilli"],
  "Assam": ["Rice", "Mustard oil", "Ginger", "Garlic", "Potato", "Spinach"],
  "Bihar": ["Wheat flour(Atta)", "Rice", "Mustard oil", "Potato", "Tomato", "Green Chilli"],
  "Chandigarh": ["Wheat flour(Atta)", "Paneer", "Tomato", "Onion", "Ghee", "Green Peas"],
  "Chhattisgarh": ["Rice", "Yellow Moong Dal", "Mustard oil", "Tomato", "Spinach", "Garlic"],
  "Dadra and Nagar Haveli and Daman and Diu": ["Rice", "Coconut", "Fish", "Tomato", "Vegetable oil"],
  "Delhi": ["Wheat flour(Atta)", "Paneer", "Tomato", "Onion", "Garlic", "Ghee"],
  "Goa": ["Rice", "Coconut milk", "Tomato", "Onion", "Garlic", "Black pepper"],
  "Gujarat": ["Besan(Gram Flour)", "Rice", "Toor Dal(Arhar)", "Peanuts(Groundnut)", "Mustard seeds(Rai)", "Cumin seeds(Jeera)"],
  "Haryana": ["Wheat flour(Atta)", "Milk", "Ghee", "Potato", "Onion", "Green Peas"],
  "Himachal Pradesh": ["Wheat flour(Atta)", "Rice", "Mustard oil", "Curd(Yogurt)", "Garlic", "Cumin seeds(Jeera)"],
  "Jammu and Kashmir": ["Rice", "Mustard oil", "Cardamom", "Cinnamon", "Fennel seeds(Saunf)", "Ginger"],
  "Jharkhand": ["Rice", "Chana Dal(Bengal Gram)", "Mustard oil", "Potato", "Tomato", "Garlic"],
  "Karnataka": ["Rice", "Toor Dal(Arhar)", "Coconut", "Mustard seeds(Rai)", "Curry leaves", "Tomato"],
  "Kerala": ["Rice", "Coconut oil", "Coconut milk", "Curry leaves", "Mustard seeds(Rai)", "Green Chilli"],
  "Ladakh": ["Barley(Jau)", "Wheat flour(Atta)", "Potato", "Butter", "Garlic"],
  "Lakshadweep": ["Rice", "Coconut oil", "Fish", "Curry leaves", "Coconut"],
  "Madhya Pradesh": ["Wheat flour(Atta)", "Poha(Beaten Rice)", "Potato", "Onion", "Garlic", "Cumin seeds(Jeera)"],
  "Maharashtra": ["Rice", "Jowar(Sorghum)", "Peanuts(Groundnut)", "Onion", "Tomato", "Mustard seeds(Rai)"],
  "Manipur": ["Rice", "Mustard oil", "Ginger", "Garlic", "Green Chilli", "Spinach"],
  "Meghalaya": ["Rice", "Ginger", "Garlic", "Mustard oil", "Black pepper", "Potato"],
  "Mizoram": ["Rice", "Mustard oil", "Ginger", "Garlic", "Green Chilli", "Spinach"],
  "Nagaland": ["Rice", "Ginger", "Garlic", "Mustard oil", "Green Chilli", "Spinach"],
  "Odisha": ["Rice", "Yellow Moong Dal", "Mustard oil", "Potato", "Tomato", "Eggplant"],
  "Puducherry": ["Rice", "Toor Dal(Arhar)", "Coconut oil", "Curry leaves", "Tomato"],
  "Punjab": ["Wheat flour(Atta)", "Paneer", "Tomato", "Onion", "Garlic", "Ghee"],
  "Rajasthan": ["Besan(Gram Flour)", "Wheat flour(Atta)", "Ghee", "Cumin seeds(Jeera)", "Onion", "Garlic"],
  "Sikkim": ["Rice", "Ginger", "Garlic", "Cardamom", "Spinach", "Mustard oil"],
  "Tamil Nadu": ["Rice", "Toor Dal(Arhar)", "Tomato", "Curry leaves", "Mustard seeds(Rai)", "Tamarind"],
  "Telangana": ["Rice", "Toor Dal(Arhar)", "Tomato", "Chili powder", "Garlic", "Onion"],
  "Tripura": ["Rice", "Mustard oil", "Ginger", "Garlic", "Potato", "Green Chilli"],
  "Uttar Pradesh": ["Wheat flour(Atta)", "Rice", "Potato", "Mustard oil", "Tomato", "Garlic"],
  "Uttarakhand": ["Rice", "Ragi(Finger Millet)", "Mustard oil", "Potato", "Garlic", "Ginger"],
  "West Bengal": ["Rice", "Mustard oil", "Potato", "Tomato", "Turmeric powder", "Cumin seeds(Jeera)"]
};

const CATEGORIZED_INGREDIENTS = {
  "Vegetables": [
    "Tomato", "Onion", "Garlic", "Ginger", "Potato", "Spinach", "Green Chilli",
    "Carrot", "Bell pepper", "Cauliflower", "Cabbage", "Green Peas", "Eggplant",
    "Bhindi(Ladies Finger)", "Bottle gourd(Lauki)", "Bitter gourd(Karela)",
    "Ridgeguard(Tori)", "Pointed gourd(Parval)", "Pumpkin", "Cucumber", "Radish",
    "Mushroom", "Sweet potato", "Beetroot", "Drumstick", "Elephant Yam(Suran)",
    "Colocasia(Arbi)", "Fenugreek leaves(Methi)", "Mint leaves(Pudina)", "Coriander leaves"
  ],
  "Pulses & Lentils": [
    "Toor Dal(Arhar)", "Yellow Moong Dal", "Moong Dal(Whole)", "Chana Dal(Bengal Gram)",
    "Urad Dal(Black Gram)", "Masur Dal(Lentil)", "Chickpeas(Kabuli Chana)", "Rajma(Kidney Beans)",
    "Cowpea(Lobia)", "Green Peas(Dry)", "Soyabean", "Horse Gram(Kulthi)"
  ],
  "Grains & Millets": [
    "Rice", "Basmati Rice", "Wheat flour(Atta)", "Maida Atta", "Poha(Beaten Rice)",
    "Besan(Gram Flour)", "Bajra(Pearl Millet)", "Jowar(Sorghum)", "Ragi(Finger Millet)",
    "Barley(Jau)", "Maize(Corn)", "Sabu Dana(Tapioca)", "Oats", "Bread", "Pasta"
  ],
  "Dairy & Plant Proteins": [
    "Paneer", "Tofu", "Soya chunks", "Milk", "Curd(Yogurt)", "Butter", "Ghee",
    "Heavy cream", "Cheese", "Cottage cheese"
  ],
  "Spices & Seasonings": [
    "Salt", "Turmeric powder", "Cumin seeds(Jeera)", "Mustard seeds(Rai)",
    "Coriander powder", "Garam masala", "Chili powder", "Black pepper",
    "Fenugreek seeds(Methi)", "Fennel seeds(Saunf)", "Ajwain(Carom)",
    "Cardamom", "Cinnamon", "Cloves", "Bay leaf(Tejpatta)", "Kasuri Methi",
    "Curry leaves", "Chaat Masala", "Amchur(Dry Mango)", "Tamarind",
    "Hing(Asafoetida)", "Saffron", "Star anise", "Nutmeg", "Paprika", "Oregano", "Basil"
  ],
  "Oils & Liquids": [
    "Mustard oil", "Coconut oil", "Groundnut oil", "Sesame oil", "Sunflower oil",
    "Vegetable oil", "Olive oil", "Lemon juice", "Soy sauce", "Vinegar", "Coconut milk"
  ],
  "Nuts & Dry Fruits": [
    "Almond(Badam)", "Cashewnuts(Kaju)", "Peanuts(Groundnut)", "Coconut",
    "Dry Grapes(Raisins)", "Walnut", "Dates", "Foxnut(Makhana)", "Sesame seeds(Til)"
  ],
  "Meat & Seafood": [
    "Egg", "Chicken", "Chicken breast", "Mutton", "Fish", "Shrimp", "Salmon", "Pork"
  ]
};
const COMMON_INGREDIENTS = Object.values(CATEGORIZED_INGREDIENTS).flat();

const UNSPLASH_CACHE = {
    "Curry": [
        "1585937421612-70a008356fbe", // Indian masala curry in handi with rice
        "1603894584373-5ac82b2ae398", // Rich gravy tikka masala in pan
        "1565557623262-b51c2513a641", // Dal makhani & curry spread with naan
        "1588166524941-3bf61a9c41db", // Aromatic tikka masala curry with basmati rice
        "1631452180519-c014fe946bc7", // Paneer butter masala handi
        "1596797038530-2c107229654b"  // Pan of vegetable masala curry with coriander
    ],
    "Chicken": [
        "1603894584373-5ac82b2ae398", // Butter chicken curry in pan
        "1588166524941-3bf61a9c41db", // Chicken tikka masala with basmati rice
        "1541832676-9b763b0239ab",    // Chicken curry with rice & lime
        "1610057099443-fde8c4d50f91", // Glazed herb roasted chicken
        "1598515214211-89d3c73ae83b", // Grilled chicken breast with vegetables
        "1604908176997-125f25cc6f3d"  // Roasted chicken skillet bowl
    ],
    "Egg": [
        "1766146431872-a4472ccd3889", // Golden masala omelette with salad & toast
        "1510693206972-df098062cb71", // Sizzling cast iron shakshuka / egg curry skillet
        "1590301157890-4810ed352733", // Egg rice bowl with fried eggs, vegetables & rice
        "1525351484163-7529414344d8"  // Sunny side up egg on whole wheat toast
    ],
    "Biryani": [
        "1563379091339-03b21ab4a4f8", // Royal saffron dum biryani platter in copper handi
        "1589302168068-964664d93dc0", // Fragrant Hyderabadi biryani with raita & mint
        "1541832676-9b763b0239ab"     // Chicken biryani rice bowl with lime
    ],
    "Salad": [
        "1540420773420-3366772f4999", // Vibrant rainbow raw vegetable salad bowl
        "1512621776951-a57141f2eefd"  // Fresh vegan buddha bowl with chickpeas & avocado
    ],
    "Dessert": [
        "1488477181946-6428a0291777", // Strawberry panna cotta dessert jars
        "1550617931-e17a7b70dce2",    // Chocolate frosted cupcakes
        "1551024601-bec78aea704b",    // Sprinkled glazed donuts
        "1563805042-7684c019e1cb"     // Oreo chocolate sundae parfait
    ],
    "Bread": [
        "1509440159596-0249088772ff", // Rustic artisan loaves of bread
        "1608198093002-ad4e005484ec"  // Freshly baked breads, buns & baguettes
    ],
    "Soup": [
        "1604152135912-04a022e23696", // Creamy yellow lentil / vegetable soup bowl
        "1547592166-23ac45744acd",    // Spiced tomato vegetable soup / shorba with seeds
        "1578020190125-f4f7c18bc9cb"  // Rich vegetable stew / tomato stew pan with bread
    ],
    "Rice_Dish": [
        "1645177628172-a94c1f96e6db", // Indian spiced vegetable pulao / tomato rice platter
        "1603133872878-684f208fb84b", // Egg / vegetable fried rice mound with peas & carrots
        "1516684732162-798a0062be99", // Steamed white rice bowl with toasted sesame
        "1563379091339-03b21ab4a4f8"  // Royal saffron basmati rice
    ],
    "Dry_Sabzi": [
        "1546833999-b9f581a1996d",    // Spiced vegetable roast / dry sabzi thali
        "1596797038530-2c107229654b", // Spiced vegetable skillet fry
        "1565557623262-b51c2513a641"  // Spiced vegetable masala with naan
    ],
    "Paneer": [
        "1631452180519-c014fe946bc7", // Rich paneer butter masala handi
        "1588166524941-3bf61a9c41db"  // Paneer tikka masala over rice
    ],
    "NonVegan": [
        "1603894584373-5ac82b2ae398",
        "1588166524941-3bf61a9c41db",
        "1610057099443-fde8c4d50f91",
        "1541832676-9b763b0239ab"
    ],
    "Vegan": [
        "1585937421612-70a008356fbe", // Indian vegetable curry handi with rice
        "1540420773420-3366772f4999", // Rainbow plant-based salad bowl
        "1512621776951-a57141f2eefd", // Fresh vegan buddha bowl with chickpeas
        "1546833999-b9f581a1996d",    // Indian vegetable thali
        "1604152135912-04a022e23696", // Creamy yellow lentil / vegetable soup bowl
        "1645177628172-a94c1f96e6db"  // Indian spiced vegetable pulao
    ]
};

const NON_VEG_WORDS = [
    "chicken", "mutton", "meat", "lamb", "beef", "pork", "bacon", "sausage", 
    "fish", "catfish", "salmon", "tuna", "cod", "prawn", "prawns", "shrimp", "shrimps",
    "seafood", "crab", "lobster", "anchovy", "egg", "eggs", "anda", "bhurji", "omelette",
    "milk", "cheese", "paneer", "butter", "ghee", "curd", "yogurt", "cream", 
    "malai", "mayo", "mayonnaise", "honey", "gelatin"
];

export const isPlantBased = (ingredients = [], recipeText = "") => {
    const raw = typeof recipeText === 'string' ? recipeText : String(recipeText || '');
    const titleMatch = raw.match(/### TITLE:\s*([^\n\r]+)/i);
    const title = (titleMatch ? titleMatch[1] : raw.split(/\r?\n/)[0]).toLowerCase();

    for (const kw of NON_VEG_WORDS) {
        const regex = new RegExp(`\\b${kw}\\b`, 'i');
        if (regex.test(title)) {
            if (kw === "milk" && /coconut milk|soy milk|almond milk|oat milk/i.test(title)) continue;
            if (kw === "cheese" && /vegan cheese/i.test(title)) continue;
            if (kw === "butter" && /peanut butter|vegan butter|almond butter/i.test(title)) continue;
            if (kw === "cream" && /coconut cream|cashew cream|vegan cream/i.test(title)) continue;
            return false;
        }
    }

    const ingList = Array.isArray(ingredients) ? ingredients : [];
    if (ingList.length > 0) {
        for (const ing of ingList) {
            const ingStr = String(ing).toLowerCase();
            for (const kw of NON_VEG_WORDS) {
                const regex = new RegExp(`\\b${kw}\\b`, 'i');
                if (regex.test(ingStr)) {
                    if (kw === "milk" && /coconut milk|soy milk|almond milk|oat milk/i.test(ingStr)) continue;
                    if (kw === "cheese" && /vegan cheese/i.test(ingStr)) continue;
                    if (kw === "butter" && /peanut butter|vegan butter|almond butter/i.test(ingStr)) continue;
                    if (kw === "cream" && /coconut cream|cashew cream|vegan cream/i.test(ingStr)) continue;
                    return false;
                }
            }
        }
    } else if (raw) {
        const lowerRaw = raw.toLowerCase();
        for (const kw of NON_VEG_WORDS) {
            const regex = new RegExp(`\\b${kw}\\b`, 'i');
            if (regex.test(lowerRaw)) {
                if (kw === "milk" && /coconut milk|soy milk|almond milk|oat milk/i.test(lowerRaw)) continue;
                if (kw === "cheese" && /vegan cheese/i.test(lowerRaw)) continue;
                if (kw === "butter" && /peanut butter|vegan butter|almond butter/i.test(lowerRaw)) continue;
                if (kw === "cream" && /coconut cream|cashew cream|vegan cream/i.test(lowerRaw)) continue;
                if (kw === "meat" && /meat texture|meat substitute|plant meat|like meat/i.test(lowerRaw)) continue;
                return false;
            }
        }
    }
    return true;
};

const BANNED_CORRUPTED_PHOTO_IDS = [
    "1618160702438", // Blueberry birthday cake falsely in Curry
    "1587314168485", // Strawberry crepes falsely in Egg
    "1565299585323", // Tacos falsely in Egg
    "1547592180",    // Salmon bowl falsely in Soup
    "1490645935",    // Salad with boiled eggs falsely in Vegan
    "1505253716",    // Shrimp salad falsely in Rice
    "1562967914",    // Chicken nuggets with tartar sauce
    "1626082927",    // Deep-fried KFC drumsticks in fryer basket
    "1578985545062"  // Old chocolate cake
];

const BANNED_MEAT_PHOTO_IDS = [
    "1563379091339", "1589302168068", "1541832676", "1598515214211", 
    "1604908176997", "1626082927389", "1562967914", "1525351484163", 
    "1565299585323", "1587314168485", "1578985545062",
    "1574484284002", "1512058564366", "1546069901", "1610057099443"
];

const SAVORY_WORDS_REGEX = /\b(curry|gravy|masala|korma|dal|dhal|daal|sambar|rasam|sabzi|subzi|roast|fry|stir-fry|sukka|poriyal|bhaji|rice|pulao|pilaf|biryani|khichdi|skillet|bowl|stew|soup|broth|shorba|chowder|kadhi|karhi|pakora|pakoras|kofta|paneer|tofu|mushroom|potato|aloo|gobi|chana|besan|chaap|tikka|kebab|roti|naan|paratha|bread|salad|chaat|omelette|egg|chicken|mutton|fish|prawn|shrimp|salt)\b/i;

const getRandomBanner = (archetype, isVegan = false, recipeTitle = "", ingredients = []) => {
    let raw = (recipeTitle || "");
    const titleMatch = raw.match(/### TITLE:\s*([^\n\r]+)/i);
    let title = titleMatch ? titleMatch[1] : raw.split(/\r?\n/)[0];
    const t = title.toLowerCase().replace(/[*_#"`]/g, '').trim();
    let arch = archetype;

    const ingsStr = Array.isArray(ingredients) ? ingredients.join(' ').toLowerCase() : String(ingredients || '').toLowerCase();
    const allContent = `${t} ${ingsStr}`;

    // 1. Animal / Poultry / Seafood / Egg / Paneer detection
    let hasChicken = /\b(chicken|mutton|meat|lamb)\b/i.test(t) || /\b(chicken|mutton|meat|lamb)\b/i.test(ingsStr);
    let hasSeafood = /\b(fish|catfish|salmon|prawn|prawns|shrimp|shrimps|seafood|pork|beef|bacon|sausage)\b/i.test(t) || /\b(fish|salmon|prawn|shrimp)\b/i.test(ingsStr);
    let hasEgg = (/\b(egg|eggs|anda|omelette)\b/i.test(t) || /\b(egg|eggs)\b/i.test(ingsStr)) && !/\b(paneer|tofu)\b/i.test(t);
    let hasPaneer = /\b(paneer)\b/i.test(t) || /\b(paneer)\b/i.test(ingsStr);

    if (isVegan || (!hasChicken && !hasSeafood && !hasEgg && !hasPaneer)) {
        hasChicken = false;
        hasSeafood = false;
        hasEgg = false;
        hasPaneer = false;
    }

    const words = t.split(/[\s\-_,]+/).filter(Boolean);
    const last1 = words[words.length - 1] || '';
    const last2 = words.slice(-2).join(' ') || last1;
    const last3 = words.slice(-3).join(' ') || last2;

    const isSavory = SAVORY_WORDS_REGEX.test(allContent);

    // A. Check for Real Dessert: ONLY sweet dishes, strictly zero savory keywords
    if (!isSavory && (/\b(cookie|cookies|crisps|cake|cakes|pastry|brownie|brownies|halwa|kheer|payasam|pudding|ice cream|barfi|ladoo|dessert)\b/i.test(last1) || /\b(cookie|cookies|crisps|cake|cakes|pastry|brownie|brownies|halwa|kheer|payasam|pudding|ice cream|barfi|ladoo|dessert)\b/i.test(last2) || /\b(cookie|cookies|crisps|cake|cakes|pastry|brownie|brownies|halwa|kheer|payasam|pudding|ice cream|barfi|ladoo|dessert)\b/i.test(t))) {
        arch = "Dessert";
    } else if (hasChicken) {
        if (/\b(biryani|rice)\b/i.test(last1) || /\b(biryani|rice)\b/i.test(last2) || /\b(biryani|rice)\b/i.test(t)) {
            arch = "Biryani";
        } else {
            arch = "Chicken";
        }
    } else if (hasSeafood) {
        arch = isVegan ? "Curry" : "NonVegan";
    } else if (hasEgg) {
        arch = "Egg";
    } else if (hasPaneer) {
        arch = "Paneer";
    } else if (/\b(stew|soup|broth|shorba|chowder|potage|laksa)\b/i.test(last1) || /\b(stew|soup|broth|shorba|chowder|potage|laksa)\b/i.test(last2)) {
        arch = "Soup";
    } else if (/\b(rice bowl|rice skillet|fried rice|pulao|pulav|pilaf|biryani|khichdi|rice|skillet)\b/i.test(last1) || /\b(rice bowl|rice skillet|fried rice|pulao|pulav|pilaf|biryani|khichdi|rice|skillet)\b/i.test(last2) || /\b(rice bowl|rice skillet|fried rice|pulao|pulav|pilaf|biryani|khichdi|rice|skillet)\b/i.test(last3) || /\b(rice bowl|rice skillet|fried rice|pulao|pulav|pilaf|biryani|khichdi|rice|skillet)\b/i.test(t)) {
        arch = "Rice_Dish";
    } else if (/\b(dal|dhal|daal|tadka|sambar|rasam)\b/i.test(last1) || /\b(dal|dhal|daal|tadka|sambar|rasam)\b/i.test(last2)) {
        arch = "Curry";
    } else if (/\b(gravy|curry|korma|masala|saag|makhani|kofta|kadhi|karhi|salan|jalfrezi|handi)\b/i.test(last1) || /\b(gravy|curry|korma|masala|saag|makhani|kofta|kadhi|karhi|salan|jalfrezi|handi)\b/i.test(last2)) {
        arch = "Curry";
    } else if (/\b(roast|fry|stir-fry|sukka|poriyal|bhaji|thoran|dry|pakora|pakoras|pakoda|pakodas|fritter|fritters|tikka|kebab|tikki|chaap|sabzi|subzi)\b/i.test(last1) || /\b(roast|fry|stir-fry|sukka|poriyal|bhaji|thoran|dry|pakora|pakoras|pakoda|pakodas|fritter|fritters|tikka|kebab|tikki|chaap|sabzi|subzi)\b/i.test(last2)) {
        arch = "Dry_Sabzi";
    } else if (/\b(roti|naan|paratha|kulcha|puri|poori|bhatura|chapati|phulka|bread|flatbread|wrap|roll)\b/i.test(last1) || /\b(roti|naan|paratha|kulcha|puri|poori|bhatura|chapati|phulka|bread|flatbread|wrap|roll)\b/i.test(last2)) {
        arch = "Bread";
    } else if (/\b(salad|slaw|chaat|raita|kachumber)\b/i.test(last1) || /\b(salad|slaw|chaat|raita|kachumber)\b/i.test(last2)) {
        arch = "Salad";
    } else {
        arch = /\brice\b/i.test(t) ? "Rice_Dish" : (isVegan ? "Vegan" : "Curry");
    }

    // Strict Savory vs. Dessert Barrier: A savory dish must NEVER have a dessert banner
    if (arch === "Dessert" && isSavory) {
        arch = isVegan ? "Vegan" : "Curry";
    }

    // Strict Vegan Protection: A vegan/plant dish must NEVER have meat/egg/biryani banners
    if (isVegan && ["Chicken", "Egg", "Biryani", "NonVegan"].includes(arch)) {
        arch = /\b(rice|biryani|pulao)\b/i.test(t) ? "Rice_Dish" : "Vegan";
    }

    // Fallback if archetype not found
    if (!UNSPLASH_CACHE[arch]) {
        arch = isVegan ? "Vegan" : "Curry";
    }

    const photos = UNSPLASH_CACHE[arch] || UNSPLASH_CACHE["Curry"];
    
    // Deterministic selection based on recipe title hash so banner never flickers on re-render
    let hash = 0;
    for (let i = 0; i < t.length; i++) {
        hash = (hash << 5) - hash + t.charCodeAt(i);
        hash |= 0;
    }
    const idx = Math.abs(hash) % photos.length;
    const photoId = photos[idx];
    return `https://images.unsplash.com/photo-${photoId}?q=80&w=1632&auto=format&fit=crop`;
};

const resolveRecipeImage = (imageUrl, archetype, isVegan, recipeText, ingredients = []) => {
    const isPlant = isVegan || isPlantBased(ingredients, recipeText);
    
    if (imageUrl) {
        // 1. Check if imageUrl has any corrupted/poisoned photo
        if (BANNED_CORRUPTED_PHOTO_IDS.some(id => imageUrl.includes(id))) {
            return getRandomBanner(archetype, isPlant, recipeText, ingredients);
        }
        
        // 2. Strict vegan barrier: plant-based recipes cannot have meat photos
        if (isPlant && BANNED_MEAT_PHOTO_IDS.some(id => imageUrl.includes(id))) {
            return getRandomBanner(archetype, true, recipeText, ingredients);
        }
        
        // 3. Strict savory barrier: savory recipes cannot have dessert photos
        const raw = recipeText || "";
        const titleMatch = raw.match(/### TITLE:\s*([^\n\r]+)/i);
        const title = (titleMatch ? titleMatch[1] : raw.split(/\r?\n/)[0]).toLowerCase();
        const isSavory = SAVORY_WORDS_REGEX.test(title);
        const isDessertImg = UNSPLASH_CACHE["Dessert"]?.some(id => imageUrl.includes(id)) || imageUrl.includes("1578985545062") || imageUrl.includes("1618160702438") || imageUrl.includes("1587314168485");
        if (isSavory && isDessertImg) {
            return getRandomBanner(isPlant ? 'Vegan' : 'Curry', isPlant, recipeText, ingredients);
        }
        return imageUrl;
    }
    return getRandomBanner(archetype, isPlant, recipeText, ingredients);
};

function MultiSelectIngredient({ selected, setSelected }) {
  const [inputValue, setInputValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = React.useRef(null);

  const trimmedInput = inputValue.trim().toLowerCase();

  const filtered = COMMON_INGREDIENTS.filter(
    ing => (trimmedInput === '' || ing.toLowerCase().includes(trimmedInput)) && !selected.includes(ing)
  ).sort((a, b) => {
    if (!trimmedInput) return 0; // No input, keep default order
    const aLower = a.toLowerCase();
    const bLower = b.toLowerCase();
    const aStarts = aLower.startsWith(trimmedInput);
    const bStarts = bLower.startsWith(trimmedInput);
    if (aStarts && !bStarts) return -1;
    if (!aStarts && bStarts) return 1;
    return a.localeCompare(b);
  }).slice(0, 8);

  const addIngredient = (ingStr) => {
    // Handle comma-separated pastes or typing
    const items = ingStr.split(',').map(i => i.trim()).filter(Boolean);
    if (items.length === 0) return;
    
    // Capitalize first letter of each word to keep it organized
    const formattedItems = items.map(item => item.charAt(0).toUpperCase() + item.slice(1).toLowerCase());
    
    const newSelected = [...selected];
    formattedItems.forEach(item => {
      // Check case-insensitive existence
      if (!newSelected.some(existing => existing.toLowerCase() === item.toLowerCase())) {
        newSelected.push(item);
      }
    });
    
    setSelected(newSelected);
    setInputValue('');
    inputRef.current?.focus();
  };

  const removeIngredient = (ing) => {
    setSelected(selected.filter(i => i !== ing));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      // If there's an exact suggestion match or user highlighted one, we'd normally pick it.
      // For now, just add whatever is in the input box if they hit enter.
      if (inputValue.trim()) {
        addIngredient(inputValue);
      }
    } else if (e.key === ',' && inputValue.trim()) {
      e.preventDefault();
      addIngredient(inputValue);
    } else if (e.key === 'Backspace' && inputValue === '' && selected.length > 0) {
      removeIngredient(selected[selected.length - 1]);
    }
  };

  const handleInputChange = (e) => {
    const val = e.target.value;
    // If user types a comma, treat it as a separator instantly
    if (val.includes(',')) {
      addIngredient(val);
    } else {
      setInputValue(val);
      setShowSuggestions(true);
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
            borderRadius: '16px', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.3rem',
            fontWeight: '500', boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
          }}>
            {ing}
            <span style={{cursor: 'pointer', opacity: 0.6}} onClick={() => removeIngredient(ing)}>×</span>
          </span>
        ))}
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)} // delay to allow clicks on dropdown
          placeholder={selected.length === 0 ? "Type ingredients (e.g. Rice, Tomato)..." : "Add more..."}
          style={{
            flex: 1, border: 'none', outline: 'none', minWidth: '150px', fontSize: '1rem', background: 'transparent'
          }}
        />
      </div>
      {/* Show categorical view when input is empty */}
      {showSuggestions && !inputValue.trim() && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, right: 0, background: '#fff',
          border: '1px solid #e5e7eb', borderRadius: '8px', marginTop: '0.5rem',
          maxHeight: '350px', overflowY: 'auto', zIndex: 50, padding: '1rem',
          boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)',
          display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '1.5rem'
        }}>
          {Object.entries(CATEGORIZED_INGREDIENTS).map(([cat, items]) => (
            <div key={cat}>
              <h4 style={{ color: '#6b7280', fontSize: '0.75rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.2rem', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{cat}</h4>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {items.map(ing => {
                  const isSelected = selected.includes(ing);
                  return (
                    <li key={ing} onMouseDown={(e) => { e.preventDefault(); isSelected ? removeIngredient(ing) : addIngredient(ing); }} 
                        style={{ 
                          fontSize: '0.85rem', padding: '0.25rem 0.5rem', cursor: 'pointer', borderRadius: '4px', 
                          color: isSelected ? '#ef4444' : '#1f2937', 
                          background: isSelected ? '#fee2e2' : 'transparent',
                          transition: 'all 0.2s', marginBottom: '2px', display: 'flex', justifyContent: 'space-between'
                        }}
                        onMouseOver={(e) => { if(!isSelected) e.target.style.background = '#f3f4f6' }} 
                        onMouseOut={(e) => { if(!isSelected) e.target.style.background = 'transparent' }}>
                      {ing} {isSelected && <span>✓</span>}
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </div>
      )}

      {/* Show search suggestions when user is typing */}
      {showSuggestions && inputValue.trim() && (
        <ul style={{
          position: 'absolute', top: '100%', left: 0, right: 0, background: '#fff',
          border: '1px solid #e5e7eb', borderRadius: '8px', marginTop: '0.5rem',
          maxHeight: '220px', overflowY: 'auto', zIndex: 50, listStyle: 'none', padding: '0.5rem 0',
          boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)'
        }}>
          {filtered.length > 0 && <li style={{ padding: '0.2rem 1rem', fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 'bold' }}>Smart Suggestions</li>}
          {filtered.map((sug, idx) => {
            const matchIndex = sug.toLowerCase().indexOf(trimmedInput);
            const beforeMatch = sug.slice(0, matchIndex);
            const matchText = sug.slice(matchIndex, matchIndex + trimmedInput.length);
            const afterMatch = sug.slice(matchIndex + trimmedInput.length);
            return (
              <li key={idx} onMouseDown={(e) => { e.preventDefault(); addIngredient(sug); }} style={{
                padding: '0.5rem 1rem', cursor: 'pointer', transition: 'background 0.2s', display: 'flex', alignItems: 'center'
              }} onMouseOver={(e) => e.target.style.background = '#f3f4f6'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
                {matchIndex >= 0 ? (
                  <span>
                    {beforeMatch}
                    <strong style={{ color: '#ef4444' }}>{matchText}</strong>
                    {afterMatch}
                  </span>
                ) : sug}
              </li>
            );
          })}
          {!filtered.some(f => f.toLowerCase() === trimmedInput) && (
             <li onMouseDown={(e) => { e.preventDefault(); addIngredient(inputValue); }} style={{
              padding: '0.5rem 1rem', cursor: 'pointer', color: '#3b82f6', borderTop: filtered.length > 0 ? '1px solid #e5e7eb' : 'none'
            }} onMouseOver={(e) => e.target.style.background = '#f3f4f6'} onMouseOut={(e) => e.target.style.background = 'transparent'}>
              + Add custom: <strong>"{inputValue.trim()}"</strong>
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

const CURATED_FEATURED_RECIPES = [
  {
    username: "chef_aditya",
    created_at: 1728800000,
    recipe: {
      recipe: "### TITLE: Royal Saffron Dum Biryani\n1. Rinse basmati rice and parboil with whole spices (cardamom, clove, bay leaf) until 70% cooked.\n2. Layer marinated spiced vegetables and aromatic saffron-infused milk in a heavy-bottomed handi.\n3. Top with golden fried onions, fresh mint, and coriander leaves.\n4. Seal the handi lid with dough and cook on low dum heat for 25 minutes.\n5. Gently fluff the grains and serve hot with cooling cucumber raita.",
      calculated_ingredients: ["250g Basmati Rice", "100g Onion", "80g Tomato", "50g Green Peas", "2g Saffron", "15g Ghee", "5g Garam Masala"],
      archetype: "Biryani",
      is_vegan: false,
      budget: 160,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_ananya",
    created_at: 1728801000,
    recipe: {
      recipe: "### TITLE: Spiced Butter Chicken Handi\n1. Marinate tender chicken pieces in thick yogurt, Kashmiri red chili, ginger-garlic paste, and lemon juice for 30 minutes.\n2. Pan-sear the chicken until lightly charred and smoky.\n3. In a pan, simmer pureed ripe tomatoes, cashews, butter, and fragrant whole spices into a silky gravy.\n4. Add the seared chicken pieces and simmer gently for 10 minutes.\n5. Finish with fresh cream, crushed kasuri methi, and serve with warm butter naan.",
      calculated_ingredients: ["200g Chicken breast", "120g Tomato", "50g Butter", "40g Cream", "20g Ginger", "15g Garlic", "5g Kasuri Methi"],
      archetype: "Chicken",
      is_vegan: false,
      budget: 185,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1604908176997-125f25cc6f3d?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_rohit",
    created_at: 1728802000,
    recipe: {
      recipe: "### TITLE: Homestyle Dhaba Egg Curry\n1. Hard-boil 4 farm eggs, peel, make light slits, and shallow-fry with turmeric and chili powder until golden.\n2. Finely dice onions and sauté in mustard oil with cumin seeds until deeply caramelized.\n3. Add ginger-garlic paste, pureed tomatoes, coriander, and garam masala; cook until oil releases.\n4. Pour in warm water to form a robust curry gravy and simmer for 5 minutes.\n5. Lower the fried eggs into the gravy, simmer for 3 minutes, and garnish with fresh cilantro.",
      calculated_ingredients: ["4 Eggs", "150g Onion", "120g Tomato", "15g Mustard oil", "10g Ginger", "5g Garam masala", "5g Turmeric"],
      archetype: "Egg",
      is_vegan: false,
      budget: 110,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1565299585323-38d6b0865b47?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "vegan_master",
    created_at: 1728803000,
    recipe: {
      recipe: "### TITLE: Agentic Crispy Tofu Tikka Masala\n1. Press firm tofu to drain excess moisture, cut into cubes, and toss with smoked paprika, nutritional yeast, and soy yogurt.\n2. Roast in oven or pan-sear until crispy and golden brown.\n3. Prepare a rich aromatic onion-tomato gravy with coconut oil, cumin, ground coriander, and turmeric.\n4. Stir in silky coconut cream to recreate the classic velvety texture without dairy.\n5. Fold in the roasted tofu cubes, simmer for 4 minutes, and finish with fresh cilantro.",
      calculated_ingredients: ["200g Firm Tofu", "140g Tomato", "80g Onion", "40g Coconut cream", "15g Coconut oil", "5g Smoked Paprika", "5g Garam Masala"],
      archetype: "Agentic Vegan",
      is_vegan: true,
      agentic_attempts: 1,
      budget: 140,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_priya",
    created_at: 1728804000,
    recipe: {
      recipe: "### TITLE: Fragrant Basmati Vegetable Pulao\n1. Wash aged long-grain basmati rice and soak in water for 20 minutes.\n2. Heat vegetable oil with cloves, cinnamon stick, cumin, and sliced onions until fragrant.\n3. Add diced carrots, green peas, and potatoes; sauté on medium heat for 3 minutes.\n4. Add drained rice, 2 cups of water, and salt; bring to a rolling boil.\n5. Cover with tight lid, reduce flame to low, and cook for 12 minutes until all water is absorbed.",
      calculated_ingredients: ["200g Basmati Rice", "70g Carrot", "60g Green Peas", "70g Potato", "50g Onion", "15g Vegetable oil", "5g Cumin"],
      archetype: "Rice_Dish",
      is_vegan: true,
      budget: 85,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1596797038530-2c107229654b?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_karan",
    created_at: 1728805000,
    recipe: {
      recipe: "### TITLE: Spiced Aloo Gobi Matar\n1. Cut fresh cauliflower into bite-sized florets and peel potatoes into small cubes.\n2. Heat mustard oil in a kadai and crackle cumin seeds with a pinch of hing.\n3. Sauté diced onions, ginger, and green chilies until translucent.\n4. Add potatoes and cauliflower with turmeric, coriander powder, and red chili; stir-fry for 5 minutes.\n5. Cover and cook on low heat until tender, adding green peas in the last 4 minutes. Garnish with cilantro.",
      calculated_ingredients: ["180g Cauliflower", "150g Potato", "60g Green Peas", "60g Onion", "15g Mustard oil", "5g Turmeric", "5g Coriander powder"],
      archetype: "Dry_Sabzi",
      is_vegan: true,
      budget: 70,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1546833999-b9f581a1996d?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_vikram",
    created_at: 1728806000,
    recipe: {
      recipe: "### TITLE: Dal Makhani Amritsari Handi\n1. Soak whole black urad dal and kidney beans overnight in abundant water.\n2. Pressure-cook with water, salt, and garlic paste until exceptionally soft and velvety.\n3. In a separate heavy handi, cook pureed tomatoes with Kashmiri chili powder and butter until glossy.\n4. Combine cooked dal with the tomato reduction and slow-simmer on lowest flame for 45 minutes.\n5. Stir in butter and fresh cream, garnish with fresh ginger juliennes.",
      calculated_ingredients: ["150g Whole Urad Dal", "40g Kidney Beans", "120g Tomato", "40g Butter", "30g Cream", "10g Ginger", "10g Garlic"],
      archetype: "Curry",
      is_vegan: false,
      budget: 95,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "vegan_master",
    created_at: 1728807000,
    recipe: {
      recipe: "### TITLE: Agentic Soy Keema Masala\n1. Rehydrate textured soy granules in warm vegetable broth with a splash of soy sauce, then squeeze dry.\n2. Sauté chopped onions, garlic, and grated ginger in vegetable oil until golden.\n3. Add pureed tomatoes, garam masala, cumin, and a dash of smoked paprika to replicate savoury depth.\n4. Fold in the rehydrated soy granules and green peas; simmer on medium flame for 8 minutes.\n5. Squeeze fresh lemon juice over the hot keema and serve with warm pav or roti.",
      calculated_ingredients: ["100g Soya granules", "120g Onion", "100g Tomato", "50g Green Peas", "15g Vegetable oil", "5g Smoked Paprika", "5g Garam Masala"],
      archetype: "Agentic Vegan",
      is_vegan: true,
      agentic_attempts: 1,
      budget: 65,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1547592166-23ac45744acd?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_tanya",
    created_at: 1728808000,
    recipe: {
      recipe: "### TITLE: Golden Egg Fried Rice\n1. Use cold cooked day-old basmati or jasmine rice for the best grain separation.\n2. Heat sesame and vegetable oil in a smoking wok; scramble 2 eggs lightly and set aside.\n3. Toss finely diced garlic, scallions, carrots, and bell peppers on high heat for 2 minutes.\n4. Add rice, soy sauce, white pepper, and toasted sesame oil; toss vigorously until fragrant.\n5. Fold back the scrambled eggs, garnish with spring onion greens, and serve immediately.",
      calculated_ingredients: ["200g Cooked Rice", "2 Eggs", "40g Bell pepper", "30g Carrot", "20g Spring onion", "15g Soy sauce", "10g Sesame oil"],
      archetype: "Rice_Dish",
      is_vegan: false,
      budget: 80,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1525351484163-7529414344d8?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_kavita",
    created_at: 1728809000,
    recipe: {
      recipe: "### TITLE: South Indian Tangy Lemon Rice\n1. Cook basmati rice until fluffy and spread on a wide plate to cool completely.\n2. Heat sesame oil in a pan; crackle mustard seeds, urad dal, chana dal, dried red chilies, and curry leaves.\n3. Add roasted peanuts, grated ginger, and turmeric powder for the vibrant yellow color.\n4. Pour freshly squeezed lemon juice with salt into the tempering.\n5. Gently toss the cooled rice into the spiced lemon tempering until evenly coated. Serve with crispy papad.",
      calculated_ingredients: ["200g Rice", "30g Peanuts", "20ml Lemon juice", "15g Sesame oil", "5g Mustard seeds", "5g Urad dal", "3g Turmeric"],
      archetype: "Rice_Dish",
      is_vegan: true,
      budget: 55,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1574484284002-952d92456975?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_rahul",
    created_at: 1728810000,
    recipe: {
      recipe: "### TITLE: Amritsari Chana Masala\n1. Soak chickpeas overnight and boil with tea bag and whole spices until tender.\n2. In a kadai, sauté finely chopped onions until deep golden brown.\n3. Add ginger-garlic paste, pureed tomatoes, dry mango powder (amchur), and chana masala.\n4. Add boiled chickpeas along with cooking liquor and simmer for 15 minutes, mashing a few for body.\n5. Temper with ghee, cumin seeds, and slit green chilies; serve with fluffy bhature.",
      calculated_ingredients: ["180g Chickpeas", "120g Onion", "100g Tomato", "15g Ghee", "10g Ginger", "5g Amchur", "5g Chana Masala"],
      archetype: "Curry",
      is_vegan: false,
      budget: 85,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?q=80&w=1632&auto=format&fit=crop"
    }
  },
  {
    username: "chef_meera",
    created_at: 1728811000,
    recipe: {
      recipe: "### TITLE: Creamy Shahi Paneer\n1. Cut fresh cottage cheese (paneer) into triangular pieces and soak in warm water for softness.\n2. Simmer onions, cashew nuts, green cardamom, and ginger in water until soft; blend into a white paste.\n3. Cook tomato puree in butter until reduced, then fold in the cashew-onion gravy.\n4. Season with saffron strands, crushed kasuri methi, and cardamom powder.\n5. Gently slide in the paneer triangles, simmer for 3 minutes, and stir in fresh cream.",
      calculated_ingredients: ["200g Paneer", "30g Cashew nuts", "100g Tomato", "60g Onion", "30g Butter", "30g Cream", "3g Cardamom"],
      archetype: "Curry",
      is_vegan: false,
      budget: 155,
      model: "qwen-2.5-32b",
      image_url: "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?q=80&w=1632&auto=format&fit=crop"
    }
  }
];

const isGroqRecipe = (doc) => {
  if (!doc || !doc.recipe) return false;
  const rec = doc.recipe;
  // Guard: Reject recipes where cost breakdown overstepped the target budget
  if (rec.cost_breakdown && rec.budget) {
    const spent = Number(rec.cost_breakdown.total_spent || rec.cost_breakdown.total_cost || 0);
    const budget = Number(rec.budget || 0);
    if (spent > 0 && budget > 0 && spent > budget + 0.5) {
      return false; // Skip overstepped budget recipe
    }
  }
  // Accept all community recipes with valid instructions or calculated ingredients
  if (typeof rec.recipe === 'string' && rec.recipe.trim().length > 15) return true;
  if (Array.isArray(rec.calculated_ingredients) && rec.calculated_ingredients.length > 0) return true;
  return false;
};

export const cleanRecipeTitle = (recipeText) => {
  if (!recipeText) return 'Delicious Dish';
  const text = typeof recipeText === 'string' ? recipeText : String(recipeText);
  const firstLine = text.split('\n')[0] || '';
  return firstLine
    .replace(/^###\s*(TITLE:)?\s*/i, '')
    .replace(/^TITLE:\s*/i, '')
    .replace(/^#+\s*/, '')
    .replace(/^\*+|\*+$/g, '')
    .replace(/^["']+|["']+$/g, '')
    .replace(/^\*+|\*+$/g, '')
    .trim() || 'Delicious Dish';
};

export const cleanRecipeInstructions = (recipeText) => {
  if (!recipeText) return [];
  const text = typeof recipeText === 'string' ? recipeText : String(recipeText);

  let rawInstructions = text;

  // 1. If explicit DIRECTIONS / INSTRUCTIONS header exists, slice after it
  const dirHeaderMatch = text.match(/(?:###\s*(?:DIRECTIONS|INSTRUCTIONS):?|\*\*(?:DIRECTIONS|INSTRUCTIONS)\*\*|DIRECTIONS:|INSTRUCTIONS:)/i);
  if (dirHeaderMatch) {
    rawInstructions = text.slice(dirHeaderMatch.index + dirHeaderMatch[0].length);
  } else {
    // Drop the first line (title)
    const lines = text.split('\n');
    rawInstructions = lines.slice(1).join('\n');
  }

  // 2. Parse lines and filter out ingredients lists, headings, and non-step junk
  const steps = [];
  const lines = rawInstructions.split('\n');
  
  let skipUntilDirections = false;
  for (let line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Check if line is an ingredients header
    if (/^(\*\*|###\s*)?(ingredients|ingredients to use):?/i.test(trimmed)) {
      skipUntilDirections = true;
      continue;
    }

    // Check if line is a directions header
    if (/^(\*\*|###\s*)?(directions|instructions|steps|method):?/i.test(trimmed)) {
      skipUntilDirections = false;
      continue;
    }

    // If we're inside an ingredients block, skip bullet points or ingredient lines
    if (skipUntilDirections) {
      if (/^[-*•]\s+/i.test(trimmed) || /^\d+(\.\d+)?\s*(g|kg|ml|tsp|tbsp|cup)/i.test(trimmed)) {
        continue;
      }
      if (/^\d+\.\s+/i.test(trimmed)) {
        skipUntilDirections = false;
      } else {
        continue;
      }
    }

    // Skip bullet points of ingredients (e.g. "- 250.0g rice")
    if (/^[-*•]\s+\d+(\.\d+)?\s*(g|kg|ml|tsp|tbsp|cup|piece)/i.test(trimmed)) {
      continue;
    }

    // Skip markdown asterisks headings like "**Directions**"
    if (/^\*\*(directions|instructions|method|ingredients)\*\*$/i.test(trimmed)) {
      continue;
    }

    // Check if it's a numbered step
    const match = trimmed.match(/^(\d+)\.\s*(.*)/);
    if (match) {
      steps.push({ num: match[1], text: match[2].replace(/^\*+|\*+$/g, '').trim() });
    } else {
      steps.push({ num: null, text: trimmed.replace(/^\*+|\*+$/g, '').trim() });
    }
  }

  return steps;
};

function App() {
  const [viewMode, setViewMode] = useState('generate'); // 'generate' | 'community' | 'history'
  const [username, setUsername] = useState('');
  
  // Form State
  const [ingredients, setIngredients] = useState(STATE_DEFAULT_INGREDIENTS['Delhi'] || ['Wheat flour(Atta)', 'Paneer', 'Tomato', 'Onion', 'Garlic', 'Ghee']);
  
  const handleStateChange = (newState) => {
    setStateName(newState);
    if (STATE_DEFAULT_INGREDIENTS[newState]) {
      setIngredients(STATE_DEFAULT_INGREDIENTS[newState]);
    }
  };
  const [budget, setBudget] = useState(150);
  const [servings, setServings] = useState(1);
  const [stateName, setStateName] = useState('Delhi');
  const [modelVersion, setModelVersion] = useState('qwen');  // 'qwen' (Guardrails) | 'qwen_budget' (Budget Only)
  const [agenticVegan, setAgenticVegan] = useState(false);
  const [communityFilter, setCommunityFilter] = useState('all'); // 'all' | 'vegan' | 'guardrails' | 'budget'
  
  // Response State
  const [loading, setLoading] = useState(false);
  const [stepMessage, setStepMessage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [saveMessage, setSaveMessage] = useState('');
  const [saveLoading, setSaveLoading] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [usernameError, setUsernameError] = useState(false);
  const [showBudgetBreakdown, setShowBudgetBreakdown] = useState(false);
  
  // History State
  const [history, setHistory] = useState([]);

  // Share Modal State
  const [shareModalRecipe, setShareModalRecipe] = useState(null);
  const [shareCopiedToast, setShareCopiedToast] = useState('');

  // Close share modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && shareModalRecipe) {
        setShareModalRecipe(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shareModalRecipe]);

  const handleCopyShareLink = (url) => {
    navigator.clipboard.writeText(url);
    setShareCopiedToast('link');
    setTimeout(() => setShareCopiedToast(''), 2500);
  };

  const handleCopyInstagramCaption = (rec) => {
    const title = cleanRecipeTitle(rec.recipe || rec.title);
    const ingredients = (rec.calculated_ingredients || []).slice(0, 5).map(i => typeof i === 'string' ? i.replace(/^\d+(\.\d+)?g\s+/i, '') : i).join(', ');
    const shareUrl = typeof window !== 'undefined' && window.location.origin ? window.location.origin : 'https://cosylab.iiitd.edu.in/ratatouille';
    const caption = `🍽️ Made this recipe: ${title}!\n\n✨ Key Ingredients: ${ingredients}\n\n🤖 Powered by Ratatouille AI - Computational Gastronomy at CoSyLaB, IIIT Delhi.\n🔗 Try it out: ${shareUrl}\n\n#RatatouilleAI #ComputationalGastronomy #IIITDelhi #FoodAI #Recipe`;
    navigator.clipboard.writeText(caption);
    setShareCopiedToast('insta');
    setTimeout(() => setShareCopiedToast(''), 2500);
  };

  // Vegan Engine State
  const [veganIngredients, setVeganIngredients] = useState('paneer, chicken');
  const [veganArchetype, setVeganArchetype] = useState('Curry');
  const [veganResult, setVeganResult] = useState(null);
  const [veganLoading, setVeganLoading] = useState(false);
  const [veganError, setVeganError] = useState('');


  // Community State
  const [globalRecipes, setGlobalRecipes] = useState(CURATED_FEATURED_RECIPES);
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
        if (data.status === 'success' && Array.isArray(data.recipes) && data.recipes.length > 0) {
          const liveGroqs = data.recipes.filter(isGroqRecipe);
          const seen = new Set();
          const merged = [];
          for (const r of [...liveGroqs, ...CURATED_FEATURED_RECIPES]) {
            const title = (r.recipe?.recipe || '').split('\n')[0].replace(/^###\s*(TITLE:)?\s*/i, '').trim().toLowerCase();
            if (title && !seen.has(title)) {
              seen.add(title);
              merged.push(r);
            }
          }
          setGlobalRecipes(merged);
        }
      } catch (err) {
        console.warn("Using offline curated recipes", err);
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
    setSaveSuccess(false);
    setUsernameError(false);
    setStepMessage('Initializing...');
    setShowBudgetBreakdown(false);

    const ingList = Array.isArray(ingredients) ? ingredients : ingredients.split(',').map(i => i.trim()).filter(i => i);

    try {
      let endpoint = agenticVegan
        ? `${BACKEND_URL}/generate-agentic-vegan`
        : (modelVersion === 'qwen_budget' 
            ? `${BACKEND_URL}/generate-fast-budget` 
            : `${BACKEND_URL}/generate-fast-qwen`);

      let payload = {
        ingredients: ingList,
        budget: parseFloat(budget),
        servings: parseInt(servings),
        state: stateName,
        archetype: agenticVegan ? "Agentic Vegan" : (modelVersion === 'qwen_budget' ? "Budget Qwen" : "Fast Qwen"),
        is_vegan: agenticVegan,
        agentic_vegan: agenticVegan,
        model_version: modelVersion
      };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned error ${response.status}: ${errorText || response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let hasResult = false;
      let hasError = false;

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
              if (data.step === 'complete' || data.step === 'final_recipe') {
                hasResult = true;
                if (data.step === 'final_recipe') {
                  const imgUrl = resolveRecipeImage(data.image_url, data.archetype, data.is_vegan, data.recipe);
                  setResult({
                    recipe: data.recipe,
                    calculated_ingredients: data.calculated_ingredients,
                    cost_breakdown: data.cost_breakdown,
                    archetype: data.archetype,
                    is_vegan: data.is_vegan,
                    agentic_attempts: data.attempts || 0,
                    judge_score: data.judge_score,
                    image_url: imgUrl,
                    budget: data.budget,
                    model: data.model,
                    latency: data.latency,
                    city: data.city
                  });
                } else {
                  const resWithBanner = { ...data.result };
                  resWithBanner.image_url = resolveRecipeImage(resWithBanner.image_url, resWithBanner.archetype, resWithBanner.is_vegan, resWithBanner.recipe);
                  setResult(resWithBanner);
                }
                setStepMessage('');
              } else if (data.step === 'error') {
                hasError = true;
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

      if (!hasResult && !hasError) {
        setError('Recipe generation ended unexpectedly. Please try again.');
      }

    } catch (err) {
      setError(err.message || 'Connection to backend failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveRecipe = async () => {
    if (!username || !username.trim()) {
      setUsernameError(true);
      return;
    }
    
    setSaveLoading(true);
    setUsernameError(false);

    try {
      const response = await fetch(`${BACKEND_URL}/save-recipe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: username.trim(),
          recipe_data: result
        })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setSaveSuccess(true);
        setSaveMessage('✅ Recipe saved successfully!');
        if (result) {
          setGlobalRecipes(prev => [
            {
              _id: data.recipe_id || String(Date.now()),
              username: username.trim(),
              created_at: Math.floor(Date.now() / 1000),
              recipe: result
            },
            ...prev
          ]);
        }
      } else {
        alert(data.message || "Failed to save recipe");
      }
    } catch (err) {
      alert("Failed to connect to database.");
    } finally {
      setSaveLoading(false);
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
        const liveGroqs = (data.recipes || []).filter(isGroqRecipe);
        const seen = new Set();
        const deduped = [];
        for (const r of liveGroqs) {
          const norm = cleanRecipeTitle(r.recipe?.recipe).toLowerCase().trim();
          if (norm && !seen.has(norm)) {
            seen.add(norm);
            deduped.push(r);
          }
        }
        setGlobalRecipes(deduped);
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
    const pool = globalRecipes.filter(isGroqRecipe);
    const candidates = pool.length > 0 ? pool : CURATED_FEATURED_RECIPES;
    const randomDoc = candidates[Math.floor(Math.random() * candidates.length)];
    const rec = randomDoc.recipe;
    const isVeg = rec.is_vegan || isPlantBased(rec.calculated_ingredients, rec.recipe);
    const imgUrl = resolveRecipeImage(rec.image_url, rec.archetype, isVeg, rec.recipe, rec.calculated_ingredients);
    setResult({
      ...rec,
      image_url: imgUrl,
      is_vegan: isVeg,
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
                Community 🌍
              </button>
            </div>
          </div>

          <div className="nav-right">
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <a 
                href="https://cosylab.iiitd.edu.in/" 
                target="_blank" 
                rel="noopener noreferrer" 
                title="CoSyLaB Portal (IIIT Delhi)" 
                className="social-icon" 
                style={{ display: 'flex', color: '#7a1d41' }}
              >
                <Globe size={22} />
              </a>
              <a 
                href="https://twitter.com/gansbags" 
                target="_blank" 
                rel="noopener noreferrer" 
                title="Prof. Ganesh Bagler on X" 
                className="social-icon" 
                style={{ display: 'flex' }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#000000"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
              </a>
              <a 
                href="https://www.linkedin.com/company/cosylab-iiit-delhi/" 
                target="_blank" 
                rel="noopener noreferrer" 
                title="CoSyLaB on LinkedIn" 
                className="social-icon" 
                style={{ display: 'flex' }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="#0A66C2"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
              </a>
              <a 
                href="https://www.facebook.com/iiitdelhi" 
                target="_blank" 
                rel="noopener noreferrer" 
                title="IIIT Delhi on Facebook" 
                className="social-icon" 
                style={{ display: 'flex' }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="#1877F2"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.469h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.469h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
              </a>
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
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.4rem', padding: '0 0.25rem', fontSize: '0.85rem' }}>
                <span style={{ color: '#6b7280' }}>
                  📍 Regional Staples: <strong style={{ color: 'var(--text-main)' }}>{stateName}</strong>
                </span>
                <button
                  type="button"
                  onClick={() => setIngredients(STATE_DEFAULT_INGREDIENTS[stateName] || ['Rice', 'Potato', 'Tomato', 'Onion'])}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#ea580c',
                    fontWeight: 600,
                    cursor: 'pointer',
                    fontSize: '0.82rem',
                    padding: '2px 6px',
                    textDecoration: 'underline'
                  }}
                  title={`Load authentic staple ingredients for ${stateName}`}
                >
                  ↺ Load {stateName} Staples
                </button>
              </div>
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
                  onChange={(e) => handleStateChange(e.target.value)}
                  style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '1rem', color: 'var(--text-main)', cursor: 'pointer' }}
                >
                  {INDIAN_STATES.map(state => (
                    <option key={state} value={state}>{state}</option>
                  ))}
                </select>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1.25rem', justifyContent: 'center', marginTop: '0', flexWrap: 'wrap', alignItems: 'center' }}>
              <label className="vegan-toggle" style={{ 
                background: agenticVegan ? 'rgba(74, 222, 128, 0.15)' : '#ffffff', 
                padding: '0.6rem 1.25rem', 
                borderRadius: '9999px', 
                boxShadow: '0 2px 6px rgba(0,0,0,0.06)', 
                border: agenticVegan ? '2px solid #22c55e' : '1px solid #e5e7eb', 
                color: agenticVegan ? '#15803d' : '#374151', 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem',
                cursor: 'pointer',
                fontWeight: agenticVegan ? '700' : '500',
                transition: 'all 0.2s ease'
              }}>
                <input 
                  type="checkbox" 
                  checked={agenticVegan} 
                  onChange={(e) => setAgenticVegan(e.target.checked)} 
                  style={{ accentColor: '#22c55e', width: '18px', height: '18px', cursor: 'pointer' }}
                />
                🌱 Agentic Vegan
              </label>

              <div className="model-toggle" style={{ background: '#ffffff', borderRadius: '9999px', boxShadow: '0 2px 6px rgba(0,0,0,0.06)', border: '1px solid #e5e7eb', overflow: 'hidden', display: 'flex' }}>
                <button 
                  type="button" 
                  className={modelVersion === 'qwen' ? 'active' : ''} 
                  onClick={() => setModelVersion('qwen')}
                  style={{ 
                    padding: '0.6rem 1.25rem',
                    border: 'none',
                    fontWeight: modelVersion === 'qwen' ? '700' : '500',
                    background: modelVersion === 'qwen' ? '#3b82f6' : 'transparent',
                    color: modelVersion === 'qwen' ? '#ffffff' : '#4b5563',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                >
                  ⚡ Groq with Guardrails
                </button>
                <button 
                  type="button" 
                  className={modelVersion === 'qwen_budget' ? 'active' : ''} 
                  onClick={() => setModelVersion('qwen_budget')}
                  style={{ 
                    padding: '0.6rem 1.25rem',
                    border: 'none',
                    fontWeight: modelVersion === 'qwen_budget' ? '700' : '500',
                    background: modelVersion === 'qwen_budget' ? '#10b981' : 'transparent',
                    color: modelVersion === 'qwen_budget' ? '#ffffff' : '#4b5563',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                >
                  💰 Groq (Budget Only)
                </button>
              </div>
            </div>

            <div className="action-buttons">
              <button onClick={handleSubmit} disabled={loading} className="btn-primary">
                Generate Recipe
              </button>
              <button onClick={handleSurpriseMe} className="btn-secondary" disabled={loading}>
                Surprise me!!
              </button>
            </div>

            {loading && (
              <div style={{ width: '100%', maxWidth: '500px', margin: '1.5rem auto 0', background: '#f3f4f6', borderRadius: '9999px', padding: '0.25rem', border: '1px solid #e5e7eb', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.05)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 0.5rem 0.25rem 0.5rem', fontSize: '0.85rem', color: 'var(--maroon)', fontWeight: 'bold' }}>
                  <span>{stepMessage || 'Processing...'}</span>
                  <span>{elapsedSeconds}s</span>
                </div>
                <div style={{ width: '100%', height: '8px', background: '#e5e7eb', borderRadius: '9999px', overflow: 'hidden' }}>
                  <div style={{ 
                    height: '100%', 
                    background: modelVersion.startsWith('qwen') ? '#3b82f6' : agenticVegan ? '#4ade80' : 'var(--primary)', 
                    width: (!stepMessage || stepMessage.includes('Initial')) ? '20%' : 
                           stepMessage.includes('Graph') ? '40%' : 
                           stepMessage.includes('Vegan') ? '60%' : 
                           stepMessage.includes('Reject') ? '75%' : 
                           stepMessage.includes('Generat') ? '80%' : '90%', 
                    transition: 'width 0.8s ease' 
                  }}></div>
                </div>
              </div>
            )}

            <div className="chips-outer-container" style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: loading ? '0.5rem' : '0.5rem', maxWidth: '800px', width: '100%' }}>
              <div className="chip highlight" onClick={handleSurpriseMe} style={{ flexShrink: 0, zIndex: 10, boxShadow: '0 4px 6px -1px rgba(250, 204, 21, 0.4)' }}>✨ Random</div>
              <div className="marquee-wrapper" style={{ marginTop: 0, flex: 1 }}>
                <div className="chips-container marquee-content">
                  {globalRecipes.filter(isGroqRecipe).slice(0, 20).map((doc, idx) => {
                     const title = cleanRecipeTitle(doc.recipe?.recipe);
                     const isVeg = doc.recipe?.is_vegan || isPlantBased(doc.recipe?.calculated_ingredients, doc.recipe?.recipe);
                     return (
                       <div key={`a-${idx}`} className="chip" onClick={() => {
                         const imgUrl = resolveRecipeImage(doc.recipe.image_url, doc.recipe.archetype, isVeg, doc.recipe.recipe, doc.recipe.calculated_ingredients);
                         setResult({...doc.recipe, image_url: imgUrl, is_vegan: isVeg, isSurprise: true});
                       }} style={{ flexShrink: 0 }}>
                         {title}
                       </div>
                     );
                  })}
                  {/* Duplicated for infinite scrolling effect */}
                  {globalRecipes.filter(isGroqRecipe).slice(0, 20).map((doc, idx) => {
                     const title = cleanRecipeTitle(doc.recipe?.recipe);
                     const isVeg = doc.recipe?.is_vegan || isPlantBased(doc.recipe?.calculated_ingredients, doc.recipe?.recipe);
                     return (
                       <div key={`b-${idx}`} className="chip" onClick={() => {
                         const imgUrl = resolveRecipeImage(doc.recipe.image_url, doc.recipe.archetype, isVeg, doc.recipe.recipe, doc.recipe.calculated_ingredients);
                         setResult({...doc.recipe, image_url: imgUrl, is_vegan: isVeg, isSurprise: true});
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
        </div>
      )}

      {result && (
            <div className="modal-overlay">
              <div className="modal-content fade-in">
                <div className="modal-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <h2 className="modal-title" style={{ margin: 0 }}>{cleanRecipeTitle(result.recipe)}</h2>
                    {result.is_vegan && (
                      <span style={{background: '#4ade80', color: '#064e3b', padding: '4px 12px', borderRadius: '16px', fontSize: '0.85rem', fontWeight: 'bold', display: 'inline-block'}}>
                        🌱 VEGAN {result.judge_score ? `(Judge: ${(result.judge_score * 100).toFixed(0)}%)` : (result.agentic_attempts !== undefined ? `(Agentic Checks: ${result.agentic_attempts})` : '')}
                      </span>
                    )}
                  </div>
                  <button className="close-btn" onClick={() => setResult(null)}><X size={24} /></button>
                </div>
                
                {(result.image_url || result.isSurprise || result.archetype) ? (
                  <div className="modal-image-container">
                    <img 
                      src={resolveRecipeImage(result.image_url, result.archetype, result.is_vegan, result.recipe)} 
                      alt="Recipe" 
                      className="modal-image" 
                      loading="lazy"
                      referrerPolicy="no-referrer"
                      onError={(e) => {
                        e.currentTarget.onerror = null;
                        e.currentTarget.src = "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=1632&auto=format&fit=crop";
                      }}
                    />
                  </div>
                ) : (
                  <div style={{ marginTop: '70px' }}></div>
                )}
                
                <div className="modal-body">
                  <div className="ingredients-col">
                    <h3>Ingredients</h3>
                    <ul className="ingredients-list">
                      {Array.isArray(result.calculated_ingredients) && result.calculated_ingredients.map((ing, idx) => (
                        <li key={idx} className="ingredient-item">
                          <ShoppingBasket className="basket-icon" size={20} />
                          <span>{ing}</span>
                        </li>
                      ))}
                    </ul>

                    {/* Collapsible Budget Breakdown (Strictly hidden when cost breakdown data is not available) */}
                    {Boolean(result.cost_breakdown && Array.isArray(result.cost_breakdown.items) && result.cost_breakdown.items.length > 0) && (
                      <div className="budget-breakdown-wrapper" style={{ marginTop: '1.25rem' }}>
                        <button
                          type="button"
                          onClick={() => setShowBudgetBreakdown(!showBudgetBreakdown)}
                          className="budget-toggle-btn"
                          style={{
                            width: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '0.65rem 0.85rem',
                            background: showBudgetBreakdown ? 'rgba(16, 185, 129, 0.2)' : 'rgba(16, 185, 129, 0.1)',
                            border: '1px solid rgba(16, 185, 129, 0.35)',
                            borderRadius: '10px',
                            color: '#10b981',
                            fontWeight: '600',
                            cursor: 'pointer',
                            fontSize: '0.88rem',
                            transition: 'all 0.2s ease',
                            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                          }}
                        >
                          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span>💰</span>
                            <span>Ingredient Cost Breakdown</span>
                            {result.cost_breakdown && (
                              <span style={{
                                fontSize: '0.75rem',
                                background: '#10b981',
                                color: '#ffffff',
                                padding: '2px 7px',
                                borderRadius: '12px',
                                fontWeight: '700'
                              }}>
                                ₹{result.cost_breakdown.total_cost} / ₹{result.budget || result.cost_breakdown.budget}
                              </span>
                            )}
                          </span>
                          <span style={{ fontSize: '0.8rem', color: '#6ee7b7' }}>
                            {showBudgetBreakdown ? '▲ Hide' : '▼ View Breakdown'}
                          </span>
                        </button>

                        {showBudgetBreakdown && result.cost_breakdown && (
                          <div
                            className="budget-breakdown-details fade-in"
                            style={{
                              marginTop: '0.6rem',
                              padding: '0.85rem',
                              background: 'rgba(17, 24, 39, 0.85)',
                              backdropFilter: 'blur(10px)',
                              border: '1px solid rgba(255, 255, 255, 0.15)',
                              borderRadius: '10px'
                            }}
                          >
                            <table style={{ width: '100%', fontSize: '0.83rem', borderCollapse: 'collapse', color: '#e5e7eb' }}>
                              <thead>
                                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.2)', textAlign: 'left', color: '#f3f4f6', background: 'rgba(255, 255, 255, 0.05)' }}>
                                  <th style={{ padding: '6px 6px', fontWeight: '600' }}>Ingredient</th>
                                  <th style={{ padding: '6px 6px', fontWeight: '600' }}>Qty (g)</th>
                                  <th style={{ padding: '6px 6px', fontWeight: '600' }}>Rate/100g</th>
                                  <th style={{ padding: '6px 6px', textAlign: 'right', fontWeight: '600' }}>Cost (₹)</th>
                                </tr>
                              </thead>
                              <tbody>
                                {result.cost_breakdown.items && result.cost_breakdown.items.map((item, idx) => (
                                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.06)' }}>
                                    <td style={{ padding: '6px 6px', textTransform: 'capitalize', color: '#f9fafb', fontWeight: '500' }}>{item.name}</td>
                                    <td style={{ padding: '6px 6px', color: '#e2e8f0' }}>{item.grams}g</td>
                                    <td style={{ padding: '6px 6px', color: '#93c5fd' }}>₹{item.rate_per_100g}</td>
                                    <td style={{ padding: '6px 6px', textAlign: 'right', fontWeight: 'bold', color: '#34d399' }}>₹{item.cost}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            <div style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              marginTop: '0.85rem',
                              paddingTop: '0.75rem',
                              borderTop: '1px solid rgba(255, 255, 255, 0.2)',
                              fontSize: '0.88rem'
                            }}>
                              <span style={{ color: '#e2e8f0', fontWeight: '500' }}>Total Spent: <strong style={{ color: '#34d399', fontSize: '0.95rem', fontWeight: '700' }}>₹{result.cost_breakdown.total_cost}</strong></span>
                              <span style={{ color: '#e2e8f0', fontWeight: '500' }}>Headroom: <strong style={{ color: '#60a5fa', fontSize: '0.95rem', fontWeight: '700' }}>₹{result.cost_breakdown.headroom}</strong></span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="instructions-col">
                    <h3>Instructions</h3>
                    <ul className="instructions-list">
                      {cleanRecipeInstructions(result.recipe).map((step, idx) => (
                        <li key={idx}>
                          {step.num ? <strong>{step.num}.</strong> : null}{' '}
                          <span>{step.text}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="modal-footer" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
                  <button 
                    className="btn-regenerate" 
                    onClick={() => { 
                      setResult(null); 
                      setSaveMessage(''); 
                      setSaveSuccess(false); 
                      setUsernameError(false); 
                    }}
                  >
                    {result.isSurprise ? "← Close Recipe" : "Regenerate Recipe"}
                  </button>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                    {saveSuccess ? (
                      <div className="save-success-badge">
                        <span style={{ fontSize: '1.15rem' }}>✓</span> Saved to Profile ({username})!
                      </div>
                    ) : (
                      <>
                        <div style={{ position: 'relative' }}>
                          <input 
                            type="text" 
                            placeholder="Enter username to save..." 
                            value={username} 
                            onChange={(e) => {
                              setUsername(e.target.value);
                              if (usernameError) setUsernameError(false);
                            }} 
                            className={usernameError ? "input-error-shake" : ""}
                            style={{ 
                              padding: '0.75rem 1rem', 
                              borderRadius: '10px', 
                              border: usernameError ? '2px solid #ef4444' : '1px solid #d1d5db', 
                              outline: 'none', 
                              fontSize: '0.95rem',
                              backgroundColor: usernameError ? '#fef2f2' : 'white',
                              minWidth: '210px',
                              transition: 'all 0.2s ease'
                            }}
                          />
                          {usernameError && (
                            <span style={{ position: 'absolute', bottom: '-19px', left: '4px', fontSize: '0.75rem', color: '#ef4444', fontWeight: '600' }}>
                              Username required to save!
                            </span>
                          )}
                        </div>
                        <button 
                          className="btn-save-recipe"
                          disabled={saveLoading}
                          onClick={handleSaveRecipe}
                        >
                          {saveLoading ? (
                            <>
                              <span className="spinner-mini"></span>
                              <span>Saving...</span>
                            </>
                          ) : (
                            <>
                              <span>💾</span>
                              <span>Save Recipe</span>
                            </>
                          )}
                        </button>
                      </>
                    )}
                    <button 
                      type="button"
                      className="btn-share-recipe"
                      onClick={() => {
                        const isVeg = result.is_vegan || isPlantBased(result.calculated_ingredients, result.recipe);
                        const bannerImg = resolveRecipeImage(result.image_url, result.archetype, isVeg, result.recipe, result.calculated_ingredients);
                        const title = cleanRecipeTitle(result.recipe);
                        setShareModalRecipe({ ...result, bannerImg, title });
                      }}
                    >
                      <Share2 size={16} />
                      <span>Share Recipe</span>
                    </button>
                  </div>
                </div>
              </div>
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
        <div className="community-section fade-in">
          <div className="community-header">
            <h1 className="title" style={{ fontSize: '2.2rem', marginBottom: '0.4rem' }}>My Recipes 📖</h1>
            <p className="subtitle" style={{ fontSize: '1rem' }}>Saved recipes and computational gastronomy dishes for your profile</p>
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginTop: '1.25rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
              <input 
                type="text" 
                placeholder="Enter username..." 
                value={username} 
                onChange={(e) => setUsername(e.target.value)} 
                onKeyDown={(e) => e.key === 'Enter' && fetchHistory()}
                style={{ padding: '0.65rem 1.2rem', borderRadius: '25px', border: '1px solid #cbd5e1', outline: 'none', width: '260px', fontSize: '0.95rem' }}
              />
              <button 
                onClick={fetchHistory} 
                style={{ padding: '0.65rem 1.6rem', borderRadius: '25px', background: '#7a1d41', color: 'white', border: 'none', cursor: 'pointer', fontWeight: 'bold', fontSize: '0.95rem', boxShadow: '0 2px 6px rgba(122, 29, 65, 0.25)' }}
              >
                Load History
              </button>
            </div>
          </div>
          
          {loading ? (
             <div className="loader" style={{ margin: '3rem auto', display: 'block' }}></div>
          ) : history.length === 0 ? (
             <div style={{ textAlign: 'center', padding: '3.5rem', color: '#64748b' }}>
               <p style={{ fontSize: '1.2rem', fontWeight: 600 }}>No recipes saved yet for "{username || 'this user'}"!</p>
               <p style={{ fontSize: '0.95rem', marginTop: '0.5rem' }}>Generate a dish in the Create tab and click "Save Recipe" to build your collection.</p>
             </div>
          ) : (
            <div className="recipe-grid">
              {history.map((doc, idx) => {
                 const rec = doc.recipe || {};
                 const title = cleanRecipeTitle(rec.recipe);
                 const isVeg = rec.is_vegan || isPlantBased(rec.calculated_ingredients, rec.recipe);
                 const bannerImg = resolveRecipeImage(rec.image_url, rec.archetype, isVeg, rec.recipe, rec.calculated_ingredients);
                 const dateStr = doc.created_at ? new Date(doc.created_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : '';
                 
                 return (
                   <div key={idx} className="cute-recipe-card">
                     <div className="card-banner-wrapper">
                       <img 
                         src={bannerImg} 
                         alt={title} 
                         className="card-banner-img"
                         loading="lazy"
                         referrerPolicy="no-referrer"
                         onError={(e) => {
                           e.currentTarget.onerror = null;
                           e.currentTarget.src = "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=1632&auto=format&fit=crop";
                         }}
                       />
                       <div className="badge-float-left">
                         {isVeg && (
                           <span className="badge-cute badge-vegan">🌱 VEGAN</span>
                         )}
                         {(rec.archetype === 'Fast Qwen' || rec.archetype === 'Groq Guardrails' || rec.final_cvs_score !== undefined) && (
                           <span className="badge-cute badge-guardrails">⚡ Guardrails</span>
                         )}
                         {(rec.archetype === 'Budget Qwen' || rec.archetype === 'Groq Budget') && (
                           <span className="badge-cute badge-budget">💰 Budget</span>
                         )}
                         {!isVeg && rec.archetype && (
                           <span className="badge-cute" style={{ background: 'rgba(122, 29, 65, 0.85)', color: '#fff' }}>
                             {rec.archetype}
                           </span>
                         )}
                       </div>
                       {rec.budget && (
                         <span className="badge-cost">₹{rec.budget} Budget</span>
                       )}
                     </div>

                     <div className="card-content">
                       <div>
                         <div className="card-meta">
                           <span className="card-date">{dateStr ? `Saved: ${dateStr}` : 'Saved Recipe'}</span>
                         </div>
                         <h3 className="card-title" title={title}>{title}</h3>
                         
                         <div className="card-ingredients-row">
                           {(rec.calculated_ingredients || []).slice(0, 4).map((ing, i) => (
                             <span key={i} className="card-ing-chip">
                               {typeof ing === 'string' ? ing.replace(/^\d+(\.\d+)?g\s+/i, '') : ing}
                             </span>
                           ))}
                           {(rec.calculated_ingredients || []).length > 4 && (
                             <span className="card-ing-chip" style={{ background: '#f1f5f9', fontWeight: 600 }}>
                               +{(rec.calculated_ingredients.length - 4)} more
                             </span>
                           )}
                         </div>
                       </div>

                       <div className="card-actions-row">
                         <button 
                           className="card-view-btn"
                           onClick={() => {
                             setResult({
                               ...rec,
                               image_url: bannerImg,
                               isSurprise: true
                             });
                           }}
                         >
                           View Recipe & Method ✨
                         </button>
                         <button
                           type="button"
                           className="card-share-btn"
                           title="Share Recipe"
                           onClick={(e) => {
                             e.stopPropagation();
                             setShareModalRecipe({ ...rec, bannerImg, title });
                           }}
                         >
                           <Share2 size={16} />
                         </button>
                       </div>
                     </div>
                   </div>
                 );
              })}
            </div>
          )}
        </div>
      )}

      {viewMode === 'community' && (
        <div className="community-section fade-in">
          <div className="community-header">
            <h1 className="title" style={{ fontSize: '2.2rem', marginBottom: '0.4rem' }}>Community Kitchen 🍲</h1>
            <p className="subtitle" style={{ fontSize: '1rem' }}>Recent High-Quality Groq AI Recipes with Live Mandi Pricing</p>
          </div>

          <div className="community-filter-bar">
            <button 
              className={`community-filter-btn ${communityFilter === 'all' ? 'active' : ''}`}
              onClick={() => setCommunityFilter('all')}
            >
              All Recipes ({globalRecipes.filter(isGroqRecipe).length})
            </button>
            <button 
              className={`community-filter-btn ${communityFilter === 'vegan' ? 'active' : ''}`}
              onClick={() => setCommunityFilter('vegan')}
            >
              🌱 Vegan ({globalRecipes.filter(isGroqRecipe).filter(d => d.recipe?.is_vegan || isPlantBased(d.recipe?.calculated_ingredients, d.recipe?.recipe)).length})
            </button>
            <button 
              className={`community-filter-btn ${communityFilter === 'guardrails' ? 'active' : ''}`}
              onClick={() => setCommunityFilter('guardrails')}
            >
              ⚡ Guardrails ({globalRecipes.filter(isGroqRecipe).filter(d => d.recipe?.archetype === 'Fast Qwen' || d.recipe?.archetype === 'Groq Guardrails' || d.recipe?.final_cvs_score !== undefined || d.recipe?.agentic_attempts !== undefined).length})
            </button>
            <button 
              className={`community-filter-btn ${communityFilter === 'budget' ? 'active' : ''}`}
              onClick={() => setCommunityFilter('budget')}
            >
              💰 Budget ({globalRecipes.filter(isGroqRecipe).filter(d => d.recipe?.archetype === 'Budget Qwen' || d.recipe?.archetype === 'Groq Budget' || (d.recipe?.budget !== undefined && d.recipe?.budget > 0)).length})
            </button>
          </div>
          
          {globalLoading ? (
             <div className="loader" style={{ margin: '3rem auto', display: 'block' }}></div>
          ) : globalRecipes.filter(isGroqRecipe).length === 0 ? (
             <div style={{ textAlign: 'center', padding: '3rem', color: '#64748b' }}>
               <p style={{ fontSize: '1.2rem', fontWeight: 600 }}>No recipes found matching this filter.</p>
             </div>
          ) : (
            <div className="recipe-grid">
              {(() => {
                const seenCardTitles = new Set();
                return globalRecipes.filter(isGroqRecipe).filter(doc => {
                  const cardTitle = cleanRecipeTitle(doc.recipe?.recipe).toLowerCase().trim();
                  if (cardTitle && seenCardTitles.has(cardTitle)) return false;
                  if (cardTitle) seenCardTitles.add(cardTitle);
                const rec = doc.recipe;
                const isVeg = rec.is_vegan || isPlantBased(rec.calculated_ingredients, rec.recipe);
                if (communityFilter === 'vegan') return isVeg;
                if (communityFilter === 'guardrails') return rec.archetype === 'Fast Qwen' || rec.archetype === 'Groq Guardrails' || rec.final_cvs_score !== undefined || rec.agentic_attempts !== undefined;
                if (communityFilter === 'budget') return rec.archetype === 'Budget Qwen' || rec.archetype === 'Groq Budget' || (rec.budget !== undefined && rec.budget > 0);
                return true;
              }).map((doc, idx) => {
                 const rec = doc.recipe;
                 const title = cleanRecipeTitle(rec.recipe);
                 const isVeg = rec.is_vegan || isPlantBased(rec.calculated_ingredients, rec.recipe);
                 const bannerImg = resolveRecipeImage(rec.image_url, rec.archetype, isVeg, rec.recipe, rec.calculated_ingredients);
                 const dateStr = doc.created_at ? new Date(doc.created_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : '';
                 
                 return (
                   <div key={idx} className="cute-recipe-card">
                     <div className="card-banner-wrapper">
                       <img 
                         src={bannerImg} 
                         alt={title} 
                         className="card-banner-img"
                         loading="lazy"
                         referrerPolicy="no-referrer"
                         onError={(e) => {
                           e.currentTarget.onerror = null;
                           e.currentTarget.src = "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=1632&auto=format&fit=crop";
                         }}
                       />
                       <div className="badge-float-left">
                         {isVeg && (
                           <span className="badge-cute badge-vegan">🌱 VEGAN</span>
                         )}
                         {(rec.archetype === 'Fast Qwen' || rec.archetype === 'Groq Guardrails' || rec.final_cvs_score !== undefined) && (
                           <span className="badge-cute badge-guardrails">⚡ Guardrails</span>
                         )}
                         {(rec.archetype === 'Budget Qwen' || rec.archetype === 'Groq Budget') && (
                           <span className="badge-cute badge-budget">💰 Budget</span>
                         )}
                       </div>
                       {rec.agentic_attempts !== undefined && rec.agentic_attempts > 0 && (
                         <div className="badge-float-right">
                           <span className="badge-cute" style={{ background: 'rgba(255,255,255,0.9)', color: '#0f172a' }}>
                             🔄 {rec.agentic_attempts} Check{rec.agentic_attempts > 1 ? 's' : ''}
                           </span>
                         </div>
                       )}
                       {rec.budget && (
                         <span className="badge-cost">₹{rec.budget} Budget</span>
                       )}
                     </div>

                     <div className="card-content">
                       <div>
                         <div className="card-meta">
                            <span className="card-date">{dateStr}</span>
                          </div>
                         <h3 className="card-title" title={title}>{title}</h3>
                         
                         <div className="card-ingredients-row">
                           {(rec.calculated_ingredients || []).slice(0, 4).map((ing, i) => (
                             <span key={i} className="card-ing-chip">
                               {ing.replace(/^\d+(\.\d+)?g\s+/i, '')}
                             </span>
                           ))}
                           {(rec.calculated_ingredients || []).length > 4 && (
                             <span className="card-ing-chip" style={{ background: '#f1f5f9', fontWeight: 600 }}>
                               +{(rec.calculated_ingredients.length - 4)} more
                             </span>
                           )}
                         </div>
                       </div>

                       <div className="card-actions-row">
                         <button 
                           className="card-view-btn"
                           onClick={() => {
                             setResult({
                               ...rec,
                               image_url: bannerImg,
                               isSurprise: true
                             });
                           }}
                         >
                           View Recipe & Method ✨
                         </button>
                         <button
                           type="button"
                           className="card-share-btn"
                           title="Share Recipe"
                           onClick={(e) => {
                             e.stopPropagation();
                             setShareModalRecipe({ ...rec, bannerImg, title });
                           }}
                         >
                           <Share2 size={16} />
                         </button>
                       </div>
                     </div>
                   </div>
                 );
              });
              })()}
            </div>
          )}
        </div>
      )}

      {/* Recipe Share Popup Modal */}
      {shareModalRecipe && (
        <div 
          className="share-modal-overlay" 
          onClick={() => setShareModalRecipe(null)}
        >
          <div 
            className="share-modal-card" 
            onClick={(e) => e.stopPropagation()}
          >
            <div className="share-modal-header">
              <h3>
                <Share2 size={20} style={{ color: '#7a1d41' }} />
                Share Recipe
              </h3>
              <button 
                className="share-modal-close"
                onClick={() => setShareModalRecipe(null)}
                title="Close"
              >
                <X size={18} />
              </button>
            </div>

            <div className="share-recipe-preview">
              <img 
                src={shareModalRecipe.bannerImg || "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=1632&auto=format&fit=crop"} 
                alt="Recipe Preview" 
                className="share-preview-thumb" 
                onError={(e) => {
                  e.currentTarget.onerror = null;
                  e.currentTarget.src = "https://images.unsplash.com/photo-1585937421612-70a008356fbe?q=80&w=1632&auto=format&fit=crop";
                }}
              />
              <div className="share-preview-info">
                <span className="share-preview-title">{shareModalRecipe.title || cleanRecipeTitle(shareModalRecipe.recipe)}</span>
                <span className="share-preview-badge">
                  {shareModalRecipe.is_vegan || isPlantBased(shareModalRecipe.calculated_ingredients, shareModalRecipe.recipe) ? '🌱 Vegan' : (shareModalRecipe.archetype || 'AI Recipe')}
                </span>
              </div>
            </div>

            <div className="share-platform-grid">
              {/* X / Twitter */}
              <a 
                href={`https://twitter.com/intent/tweet?text=${encodeURIComponent(`Check out this recipe: "${shareModalRecipe.title || cleanRecipeTitle(shareModalRecipe.recipe)}" generated with Ratatouille AI! 🍳 #ComputationalGastronomy #IIITDelhi`)}&url=${encodeURIComponent(typeof window !== 'undefined' && window.location.origin ? window.location.origin : 'https://cosylab.iiitd.edu.in/ratatouille')}`}
                target="_blank" 
                rel="noopener noreferrer" 
                className="share-platform-btn share-btn-x"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="#ffffff"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                <span>Twitter / X</span>
              </a>

              {/* LinkedIn */}
              <a 
                href={`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(typeof window !== 'undefined' && window.location.origin ? window.location.origin : 'https://cosylab.iiitd.edu.in/ratatouille')}`}
                target="_blank" 
                rel="noopener noreferrer" 
                className="share-platform-btn share-btn-linkedin"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="#ffffff"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                <span>LinkedIn</span>
              </a>

              {/* Facebook */}
              <a 
                href={`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(typeof window !== 'undefined' && window.location.origin ? window.location.origin : 'https://cosylab.iiitd.edu.in/ratatouille')}`}
                target="_blank" 
                rel="noopener noreferrer" 
                className="share-platform-btn share-btn-facebook"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="#ffffff"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.469h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.469h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
                <span>Facebook</span>
              </a>

              {/* WhatsApp */}
              <a 
                href={`https://api.whatsapp.com/send?text=${encodeURIComponent(`Check out this recipe: *${shareModalRecipe.title || cleanRecipeTitle(shareModalRecipe.recipe)}* generated with Ratatouille AI!\n${typeof window !== 'undefined' && window.location.origin ? window.location.origin : 'https://cosylab.iiitd.edu.in/ratatouille'}`)}`}
                target="_blank" 
                rel="noopener noreferrer" 
                className="share-platform-btn share-btn-whatsapp"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="#ffffff"><path d="M.057 24l1.687-6.163c-1.041-1.804-1.588-3.849-1.587-5.946.003-6.556 5.338-11.891 11.893-11.891 3.181.001 6.167 1.24 8.413 3.488 2.245 2.248 3.481 5.236 3.48 8.414-.003 6.557-5.338 11.892-11.893 11.892-1.99-.001-3.951-.5-5.688-1.448l-6.305 1.654zm6.597-3.807c1.676.995 3.276 1.591 5.392 1.592 5.448 0 9.886-4.434 9.889-9.885.002-5.462-4.415-9.89-9.881-9.892-5.452 0-9.887 4.434-9.889 9.884-.001 2.225.651 3.891 1.746 5.634l-.999 3.648 3.742-.981zm11.387-5.464c-.074-.124-.272-.198-.57-.347-.297-.149-1.758-.868-2.031-.967-.272-.099-.47-.149-.669.149-.198.297-.768.967-.941 1.165-.173.198-.347.223-.644.074-.297-.149-1.255-.462-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.297-.347.446-.521.151-.172.2-.296.3-.495.099-.198.05-.372-.025-.521-.075-.148-.669-1.611-.916-2.206-.242-.579-.487-.501-.669-.51l-.57-.01c-.198 0-.52.074-.792.372s-1.04 1.016-1.04 2.479 1.065 2.876 1.213 3.074c.149.198 2.095 3.2 5.076 4.487.709.306 1.263.489 1.694.626.712.226 1.36.194 1.872.118.571-.085 1.758-.719 2.006-1.413.248-.695.248-1.29.173-1.414z"/></svg>
                <span>WhatsApp</span>
              </a>

              {/* Instagram Copy */}
              <button 
                type="button"
                className="share-platform-btn share-btn-instagram"
                onClick={() => handleCopyInstagramCaption(shareModalRecipe)}
              >
                <span>📸 Copy Instagram Caption / Story</span>
              </button>
            </div>

            <div className="share-copy-box">
              <span className="share-copy-label">Direct Recipe Link</span>
              <div className="share-copy-input-row">
                <input 
                  type="text" 
                  readOnly 
                  value={typeof window !== 'undefined' && window.location.origin ? window.location.origin : 'https://cosylab.iiitd.edu.in/ratatouille'} 
                  className="share-copy-input"
                  onClick={(e) => e.target.select()}
                />
                <button 
                  type="button"
                  className={`share-copy-btn ${shareCopiedToast === 'link' ? 'copied' : ''}`}
                  onClick={() => handleCopyShareLink(typeof window !== 'undefined' && window.location.origin ? window.location.origin : 'https://cosylab.iiitd.edu.in/ratatouille')}
                >
                  {shareCopiedToast === 'link' ? (
                    <>
                      <Check size={15} />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <LinkIcon size={15} />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {shareCopiedToast === 'insta' && (
              <div className="share-toast">
                <Check size={16} style={{ color: '#10b981' }} />
                <span>Instagram caption copied to clipboard!</span>
              </div>
            )}
            {shareCopiedToast === 'link' && (
              <div className="share-toast">
                <Check size={16} style={{ color: '#10b981' }} />
                <span>Link copied to clipboard!</span>
              </div>
            )}
          </div>
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
