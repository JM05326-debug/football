// 球隊英文名稱 -> 繁體中文名稱對照表
// key 必須與 model_data.js / fixtures_data.js 裡的球隊 key（英文）完全一致，
// 內部計算一律使用英文 key，這份對照表只用於畫面顯示。
// 找不到對照的球隊（多半是剛升班、還沒收錄的球隊）會直接顯示英文原名。

const TEAM_NAME_ZH = {
  // ---- 英超 Premier League ----
  "Arsenal": "阿森納",
  "Aston Villa": "阿斯頓維拉",
  "Bournemouth": "伯恩茅斯",
  "Brentford": "布倫特福",
  "Brighton": "布萊頓",
  "Burnley": "伯恩利",
  "Chelsea": "切爾西",
  "Coventry": "考文垂",
  "Crystal Palace": "水晶宮",
  "Everton": "埃弗頓",
  "Fulham": "富勒姆",
  "Hull": "赫爾城",
  "Ipswich": "伊普斯維奇",
  "Leeds": "里茲聯",
  "Leicester": "萊斯特城",
  "Liverpool": "利物浦",
  "Luton": "盧頓",
  "Man City": "曼城",
  "Man United": "曼聯",
  "Newcastle": "紐卡索聯",
  "Norwich": "諾維奇",
  "Nott'm Forest": "諾丁漢森林",
  "Sheffield United": "謝菲爾德聯",
  "Southampton": "南安普敦",
  "Sunderland": "桑德蘭",
  "Tottenham": "熱刺",
  "Watford": "沃特福",
  "West Brom": "西布羅姆維奇",
  "West Ham": "西漢姆聯",
  "Wolves": "狼隊",
};

// 找不到中文對照時，直接回傳英文原名
function zhName(englishName) {
  return TEAM_NAME_ZH[englishName] || englishName;
}
