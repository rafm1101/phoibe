# T-RIX Visualisierung — Implementierungsstrategie

---
type: plan
id: "0002"
title: "Visualisation"
status: draft
date: 2026-10-23
superseded_by: null
---

## Kontext-Präzisierung

**TRIX ist komplexer als initial angenommen:**
```
trix(site_i, ref_j) = f(rix_site_i, rix_ref_j, elev_site_i, elev_ref_j)
A(site_i, ref_j) = g(trix(site_i, ref_j))                    # Pro Pairing!
transfer(site_i, ref_j) = h(A(site_i, ref_j), dist(site_i, ref_j))
```

**Konsequenzen:**
- A ist **nicht global**, sondern **pro Pairing unterschiedlich**
- Decision Boundaries sind damit zwangsweise komplex (Szenario 4)
- Per-Pairing A-Wert ist kritisches Metadatum (nicht nur TRIX)

**Rechenbudget ist großzügig:**
- 25.000 Gitterpunkte × 100ms/Punkt ≈ 42 min/Reference
- Parallelisierbar auf 8 Cores → ~5 min/Reference (akzeptabel)
- 7 gemessene Sites sind negligible

---

## Problem: Keine konkrete Zielgruppe yet

Das ist zentral. Verschiedene Personas haben völlig unterschiedliche Bedürfnisse:

### **Persona A: Messfeldplaner** 🗺️
*"Ich plane eine Messkampagne. Wo baue ich die Instrumente hin?"*

- **Input:** DEM + vorhandene References
- **Frage:** "Wo würde ein neuer Site optimal passen?"
- **Nutzt:** Coverage Heatmap pro Reference + Coverage Mosaic
- **Interaktion:** Click → TRIX-Vorschau live
- **Output:** Karte mit Planung + Begründung

**Bedarf an Visualisierung:**
- ✅ Coverage-Flächen (Klasse 0/1/2)
- ✅ Lücken-Analyse ("Diese Region ist unterversorgt")
- ✅ Robustheit ("Wie viele References decken einen Punkt ab?")
- ✅ Live-Planung (interaktiv)
- ❌ Historische Pairings

---

### **Persona B: Analyseverfeiner** 📊
*"Ich habe 7 Sites gemessen. Welche References sind am besten? Wie kann ich das defensible begründen?"*

- **Input:** Gemessene Sites + References
- **Frage:** "Welche Ref für welchen Site? Wie nah sind die Grenzfälle?"
- **Nutzt:** Connection Diagram + Metadaten
- **Interaktion:** Hover → Details (TRIX, A, Distance, Sicherheitsmargen)
- **Output:** Begründete Klassifizierung für Report

**Bedarf an Visualisierung:**
- ✅ Paarings-Netzwerk (Site ↔ Reference)
- ✅ Grenzfall-Flagging (welche sind nahe A?)
- ✅ Pro-Pairing Metadaten (TRIX, A, Distance)
- ✅ Stabilität (wie weit weg von Grenze?)
- ❌ Gitter / Planung

---

### **Persona C: Stakeholder/QA-Reviewer** 👤
*"Ist diese Klassifizierung verteidigbar? Gibt es systematische Fehler?"*

- **Input:** Finished Analysis (Summary, Detail, TRIX)
- **Frage:** "Was sind die Schwachstellen? Wo sind die Annahmen fragil?"
- **Nutzt:** Sensitivity Report + Uncertainty Visualization
- **Interaktion:** Read-only Inspection
- **Output:** Audit-Trail

**Bedarf an Visualisierung:**
- ✅ Sensitivity (Tornado chart für Parameter)
- ✅ Uncertainty Bands (Confidence Intervals)
- ✅ Provenance (Git-Hash, DEM-Fingerprint)
- ✅ Vergleich gegen Historische Runs
- ❌ Planung / Interaktivität

---

## Modulare Umsetzungsstrategie

Statt monolithisch: **Baue separate, austauschbare Module pro Persona + eine gemeinsame Datenschicht.**

```
Data Layer (gemeinsam)
├─ ResultSummary (existiert)
├─ Extended Metadata (neu)
│  ├─ per_pairing_A: dict  # A(site_i, ref_j)
│  ├─ per_pairing_distance_to_A: dict
│  └─ per_pairing_distance_to_B: dict
└─ Coverage Grids (für B/C optional)
   ├─ reference_A/coverage.tif
   ├─ reference_B/coverage.tif
   └─ ...

Visualization Layer
├─ Module A: Connection Diagram (Persona B)
│  └─ Interactive QGIS Layer + HTML Widget
│
├─ Module B: Coverage Heatmap (Persona A)
│  ├─ Per-Reference GeoPackage + Raster
│  └─ Coverage Mosaic (optional)
│
├─ Module C: Sensitivity & Uncertainty (Persona C)
│  └─ HTML Report + interaktive Grafiken
│
└─ Module D: Site-Placement Optimizer (Persona A)
   └─ Voila Widget (interaktiv)
```

---

## Phase 1: MVP — Datenebene + Persona B

### **Schritt 1.1: Extended Metadata berechnen & speichern**

In `analyzer.py` / `results.py`:

```python
@dataclasses.dataclass
class PairingMetadata:
    """Pro (site_id, ref_id) Pairing"""
    site_id: str
    ref_id: str
    trix: float
    distance_m: float
    rix_site: float
    rix_ref: float
    elev_site: float
    elev_ref: float
    A: float                    # ← NEU: Pro Pairing A-Wert
    B: float                    # ← NEU: Pro Pairing B-Wert
    transferability: int        # 0, 1 oder 2
    distance_to_A: float        # TRIX-Distanz zu Grenze A
    distance_to_B: float        # TRIX-Distanz zu Grenze B
    stability_score: float      # [0, 1] — wie stabil ist diese Klassifizierung?
                               # 0 = direkt auf Grenze, 1 = weit weg

@dataclasses.dataclass
class ResultSummary:
    # ... (existierende Felder)
    pairings: list[PairingMetadata]  # ← NEU: Alle Pairings mit Metadaten
```

**Berechnung `stability_score`:**
```python
def compute_stability(distance_to_threshold, threshold_buffer=0.05):
    """
    distance_to_threshold: Wie nah an der Grenze bin ich?
    threshold_buffer: Sicherheitspuffer (±5% als Default)
    
    Returns: [0, 1]
    - 0.0 = direkt auf Grenze (sehr fragil)
    - 0.5 = auf Halfway zur nächsten Grenze
    - 1.0 = weit weg (stabil)
    """
    if distance_to_threshold < threshold_buffer:
        return distance_to_threshold / threshold_buffer * 0.5  # [0, 0.5]
    else:
        return 0.5 + 0.5 * min(1.0, distance_to_threshold / (threshold_buffer * 2))
```

**Speichern in `summary.yaml`:**
```yaml
pairings:
  - site_id: "Site_1"
    ref_id: "Ref_A"
    trix: 0.082
    distance_m: 12300
    A: 0.149           # Pro Pairing!
    B: 0.347
    transferability: 2
    distance_to_A: 0.067  # |0.082 - 0.149|
    distance_to_B: 0.265
    stability_score: 0.92  # Nah bei Klasse 2, aber stabil
```

---

### **Schritt 1.2: GeoPackage erweitern**

```python
def write_pairings_layer(result: ResultSummary, gpkg_path: str):
    """
    Erstelle Layer mit per-Pairing Informationen
    """
    geometries = []
    records = []
    
    for pairing in result.pairings:
        # LineString vom Site zum Reference
        site_pt = result.summary[result.summary['site_id'] == pairing.site_id].iloc[0]
        ref_pt = result.summary[result.summary['site_id'] == pairing.ref_id].iloc[0]
        
        line = LineString([
            (site_pt['geometry'].x, site_pt['geometry'].y),
            (ref_pt['geometry'].x, ref_pt['geometry'].y)
        ])
        
        records.append({
            'site_id': pairing.site_id,
            'ref_id': pairing.ref_id,
            'transferability': pairing.transferability,
            'trix': pairing.trix,
            'distance_m': pairing.distance_m,
            'A': pairing.A,                    # ← Pro Pairing!
            'B': pairing.B,
            'distance_to_A': pairing.distance_to_A,
            'distance_to_B': pairing.distance_to_B,
            'stability_score': pairing.stability_score,
            'geometry': line
        })
    
    gdf = gpd.GeoDataFrame(records, crs=result.crs)
    gdf.to_file(gpkg_path, layer='pairings', driver='GPKG')
```

---

### **Schritt 1.3: QGIS-Styling & interaktive Tabelle**

**GeoPackage Layer Styling (QGIS):**

```qgis
Layer: pairings (LineString)

Symbology:
├─ Data-defined Size & Color
│
├─ Size: stability_score
│  └─ 0.2 mm bis 2.0 mm → je stabiler, desto dicker
│
├─ Color: transferability
│  ├─ 0 → Rot (#FF4444)
│  ├─ 1 → Orange (#FFAA44)
│  └─ 2 → Grün (#44FF44)
│
├─ Dash Pattern: conditional
│  ├─ stability_score < 0.5 → Gestrichelt (fragil)
│  ├─ sonst → Solid
│
└─ Label: site_id + " → " + ref_id

Popup (on Click):
├─ Site: {{site_id}}
├─ Reference: {{ref_id}}
├─ Transferability: {{transferability}} 
│  └─ Color-coded
├─ TRIX: {{trix | round(3)}}
├─ Distance: {{distance_m | round(0)}} m
├─ A (Local): {{A | round(3)}}
├─ B (Local): {{B | round(3)}}
├─ Distance to A: {{distance_to_A | round(3)}} ⚠️
├─ Stability: {{stability_score | round(2)}}
│  └─ Green bar [========>  ] 92%
└─ Recommendation: 
   "✅ Klasse 2: Stabil & weit weg von Grenzen"
   oder
   "⚠️ Klasse 1: Nah an A-Grenze (67 TRIX-Punkte). Erhöhte Unsicherheit."
```

**Interaktive HTML-Tabelle (in Notebook / HTML-Export):**

```python
def create_pairings_table_html(result: ResultSummary) -> str:
    """Sortierbar, filterbar, farblich codiert"""
    df = pd.DataFrame([asdict(p) for p in result.pairings])
    
    # Sortierung nach stability_score (absteigend)
    df = df.sort_values('stability_score', ascending=False)
    
    # HTML mit Styling
    html = """
    <table style="border-collapse: collapse; width: 100%;">
    <tr>
        <th>Site</th>
        <th>Reference</th>
        <th>Class</th>
        <th>TRIX</th>
        <th>A (Local)</th>
        <th>Distance to A</th>
        <th>Stability</th>
        <th>Recommendation</th>
    </tr>
    """
    
    for _, row in df.iterrows():
        class_color = {0: '#FF4444', 1: '#FFAA44', 2: '#44FF44'}[row['transferability']]
        
        stability_bar = '█' * int(row['stability_score'] * 20) + '░' * (20 - int(row['stability_score'] * 20))
        
        recommendation = "✅ Sicher" if row['stability_score'] > 0.8 else "⚠️ Grenzfall"
        
        html += f"""
        <tr>
            <td>{row['site_id']}</td>
            <td>{row['ref_id']}</td>
            <td style="background-color: {class_color};">{row['transferability']}</td>
            <td>{row['trix']:.3f}</td>
            <td>{row['A']:.3f}</td>
            <td style="color: {'red' if row['distance_to_A'] < 0.05 else 'black'};"><b>{row['distance_to_A']:.3f}</b></td>
            <td><code>{stability_bar}</code> {row['stability_score']:.2f}</td>
            <td>{recommendation}</td>
        </tr>
        """
    
    html += "</table>"
    return html
```

**Ausgabe:** Tabelle, in der sofort sichtbar ist:
- Welche Pairings sind fragil (nah an Grenze)
- Welche sind stabil (weit weg)
- Pro-Pairing A-Werte zeigen Variabilität

---

## Phase 2: Coverage Heatmap (Persona A — Messfeldplanung)

### **Schritt 2.1: Gridded RIX-Berechnung**

```python
class CoverageAnalyzer:
    def __init__(self, dem: xr.DataArray, analyzer: RIXAnalyzer):
        self.dem = dem
        self.analyzer = analyzer
    
    def compute_coverage_grid(
        self,
        reference_id: str,
        rix_reference: float,
        grid_spacing_m: float = 30,
        bbox: Optional[tuple] = None  # (xmin, ymin, xmax, ymax) optional
    ) -> xr.Dataset:
        """
        Für jedes Gitterpunkt: RIX → TRIX → Klasse
        
        Rechenplan:
        - Gitter mit ~30m Spacing über 15×15km
        - ~25.000 Punkte
        - 100ms/Punkt → ~42 min single-threaded
        - Mit 8 Cores: ~5 min
        """
        
        # 1. Gitter erstellen
        grid_x, grid_y = self._create_grid(grid_spacing_m, bbox)
        
        # 2. RIX für jeden Gitterpunkt (parallelisiert)
        rix_grid = self._compute_rix_batch(grid_x, grid_y)  # xarray.DataArray
        
        # 3. TRIX für jeden Punkt
        trix_grid = xr.apply_ufunc(
            lambda rix_pt: compute_trix(
                rix_pt, 
                rix_reference, 
                elev_site=...,  # Aus DEM am Gitterpunkt
                elev_ref=...,   # Aus DEM bei Reference
                ...
            ),
            rix_grid,
            dask='parallelized',  # ← Dask für Parallelisierung
            output_dtypes=[float]
        )
        
        # 4. A und B pro Gitterpunkt (diese sind jetzt lokal)
        A_grid = xr.apply_ufunc(
            lambda t: compute_trix_limit_A(t),
            trix_grid,
            dask='parallelized',
            output_dtypes=[float]
        )
        
        B_grid = xr.apply_ufunc(
            lambda t: compute_trix_limit_B(t),
            trix_grid,
            dask='parallelized',
            output_dtypes=[float]
        )
        
        # 5. Klassifizieren
        distance_grid = self._distance_to_reference_grid(grid_x, grid_y, reference_id)
        classes = self._classify_grid(trix_grid, A_grid, B_grid, distance_grid)
        
        # 6. Output Dataset
        return xr.Dataset({
            'trix': trix_grid,
            'rix': rix_grid,
            'A': A_grid,
            'B': B_grid,
            'distance_m': distance_grid,
            'class': classes,
            'distance_to_A': (trix_grid - A_grid).abs(),
            'distance_to_B': (trix_grid - B_grid).abs()
        }, coords={'x': grid_x, 'y': grid_y})

def _compute_rix_batch(self, grid_x, grid_y) -> xr.DataArray:
    """
    Nutze RIXAnalyzer.run() mit Batch von Grid-Punkten
    oder parallelisiere über Dask
    """
    # Option 1: Mit Dask
    import dask.array as da
    
    rix_list = da.from_delayed_many(
        [delayed(self.analyzer.run_single_point)(self.dem, pt) 
         for pt in zip(grid_x.flatten(), grid_y.flatten())],
        shape=(len(grid_x.flatten()),),
        dtype=float
    )
    
    return rix_list.reshape(grid_x.shape)
```

---

### **Schritt 2.2: GeoPackage + Raster-Export**

```python
def write_coverage_output(coverage: xr.Dataset, reference_id: str, output_dir: Path):
    """
    Pro Reference: Multi-Output
    """
    
    # A) Raster (GeoTIFF)
    coverage['class'].rio.to_raster(
        output_dir / f"{reference_id}_class.tif"
    )
    coverage['trix'].rio.to_raster(
        output_dir / f"{reference_id}_trix.tif"
    )
    coverage['distance_to_A'].rio.to_raster(
        output_dir / f"{reference_id}_margin_to_A.tif"
    )
    
    # B) Isokontur-Polygone (GeoPackage)
    gpkg = output_dir / f"{reference_id}_coverage.gpkg"
    
    for cls in [0, 1, 2]:
        class_mask = coverage['class'] == cls
        polygons = rasterio.features.shapes(
            class_mask.values.astype('uint8'),
            transform=coverage.rio.transform()
        )
        
        gdf = gpd.GeoDataFrame(
            [{'class': cls, 'geometry': shape(geom)} 
             for geom, val in polygons if val > 0],
            crs=coverage.rio.crs
        )
        gdf.to_file(gpkg, layer=f'class_{cls}', driver='GPKG')
    
    # C) Attribute Table (verloren im Raster, nützlich für Analyse)
    coverage_stats = xr.Dataset({
        'mean_trix': coverage['trix'].mean(['x', 'y']),
        'std_trix': coverage['trix'].std(['x', 'y']),
        'n_class_2': (coverage['class'] == 2).sum(),
        'n_class_1': (coverage['class'] == 1).sum(),
        'n_class_0': (coverage['class'] == 0).sum(),
        'pct_class_2': 100 * (coverage['class'] == 2).mean(),
        'mean_distance_to_A': coverage['distance_to_A'].mean(),
        'n_points_fragile': (coverage['distance_to_A'] < 0.05).sum()
    })
    
    coverage_stats.to_netcdf(output_dir / f"{reference_id}_stats.nc")
```

---

### **Schritt 2.3: QGIS-Kartensatz**

**Pro Reference ein Kartensatz:**

```
reference_A_coverage.gpkg:
├─ class_2 (Polygon) → Grün, Label "Optimal"
├─ class_1 (Polygon) → Orange, Label "Bedingt"
├─ class_0 (Polygon) → Rot, Label "Nicht transferierbar"
├─ attribute_table (Table) — Meta über diese Reference
└─ metadata.yaml — Grid-Abstand, A/B-Werte, Rechenzeit
```

**QGIS-Projekt (QGZ) mit Layers:**

```
Layers (von unten nach oben)
├─ DEM (Basiskarte, semitranskparent)
├─ class_2_Reference_A (80% Opazität)
├─ class_1_Reference_A (80% Opazität)
├─ class_0_Reference_A (80% Opazität)
├─ Reference_A (Punkt, großer Star)
├─ Sites_actual (Punkte, gefärbt nach beste Ref)
└─ Grid_spacing_30m (optional, für Kontrolle)

Interaktion:
├─ Click auf Polygon → Klasse, mean TRIX, n_points, stability
├─ Toggle Layer → Focus auf einzelne Klasse
└─ Toggle "actual Sites" → Sehen, wo die Messungen tatsächlich sind vs. optimal
```

**Nutzen für Messfeldplaner:**
- "Diese 5×5km Fläche ist grün (Klasse 2) — hier würde ein Site perfekt passen"
- "Diese Region ist rot (Klasse 0) — nicht abgedeckt, könnten problematisch werden"
- "Meine 7 gemessenen Sites sind (hier), aber optimal wären sie (dort)"

---

## Phase 3: Site-Placement Optimizer (Persona A — Interaktiv)

### **Voila-Widget:**

```python
class SitePlacementOptimizer:
    """Interaktive Planung: Click auf Karte → Live TRIX Feedback"""
    
    def __init__(self, dem: xr.DataArray, references_rix: dict):
        self.dem = dem
        self.refs = references_rix  # {'Ref_A': 0.12, 'Ref_B': 0.15, ...}
        self.analyzer = RIXAnalyzer.from_config(...)
        self.candidate_rix = None
    
    def build_dashboard(self):
        # 1. Karte (Folium)
        self.map_widget = folium.Map(location=[51.0, 11.0], zoom_level=10)
        
        # References als große Marker
        for ref_id, rix in self.refs.items():
            folium.Marker(
                location=self.refs_locations[ref_id],
                popup=f"{ref_id}: RIX={rix:.3f}",
                icon=folium.Icon(color='blue', icon='star')
            ).add_to(self.map_widget)
        
        # Draw-Tool: Nutzer klickt
        draw = folium.LatLngPopup()
        draw.add_to(self.map_widget)
        
        # 2. Result-Panel
        self.result_panel = widgets.HTML(
            value="<p>Klicken Sie auf die Karte, um einen Kandidat-Site zu evaluieren.</p>"
        )
        
        # 3. On Click → Berechnung
        def on_location_selected(data):
            if data['type'] == 'click':
                lat, lng = data['coordinates']['lat'], data['coordinates']['lng']
                self._evaluate_candidate(lat, lng)
        
        draw.on_location_select(on_location_selected)
        
        return widgets.VBox([self.map_widget, self.result_panel])
    
    def _evaluate_candidate(self, lat: float, lng: float):
        """Berechne TRIX zu allen References für diesen Punkt"""
        
        # 1. RIX an diesem Punkt
        self.candidate_rix = self.analyzer.run(self.dem, [{'lat': lat, 'lng': lng}])
        
        # 2. TRIX zu allen References
        results = []
        for ref_id, rix_ref in self.refs.items():
            trix = compute_trix(self.candidate_rix, rix_ref, ...)
            A = compute_trix_limit_A(trix)
            cls = classify(trix, A, ...)
            
            results.append({
                'reference': ref_id,
                'rix_site': self.candidate_rix,
                'rix_ref': rix_ref,
                'trix': trix,
                'A': A,
                'class': cls,
                'color': {0: '🔴', 1: '🟠', 2: '🟢'}[cls]
            })
        
        # 3. Sortiere nach Klasse (beste zuerst)
        results.sort(key=lambda x: x['class'], reverse=True)
        
        # 4. Update Panel
        html = f"""
        <h3>Site-Kandidat evaluiert</h3>
        <p><b>Position:</b> {lat:.4f}, {lng:.4f}</p>
        <p><b>RIX:</b> {self.candidate_rix:.3f}</p>
        
        <h4>Empfehlung pro Reference:</h4>
        <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <th>Reference</th>
            <th>RIX (Ref)</th>
            <th>TRIX</th>
            <th>A (Local)</th>
            <th>Class</th>
            <th>Best?</th>
        </tr>
        """
        
        for i, r in enumerate(results):
            best = "⭐ BESTE" if i == 0 else ""
            html += f"""
            <tr>
                <td><b>{r['reference']}</b></td>
                <td>{r['rix_ref']:.3f}</td>
                <td>{r['trix']:.3f}</td>
                <td>{r['A']:.3f}</td>
                <td>{r['color']} Klasse {r['class']}</td>
                <td>{best}</td>
            </tr>
            """
        
        html += """
        </table>
        <p style="font-size: 12px; color: #666;">
        Hinweis: Dies ist eine Schnell-Evaluierung. Für finale Entscheidung 
        siehe auch Coverage Heatmaps pro Reference.
        </p>
        """
        
        self.result_panel.value = html
```

**Voila-Start:**
```bash
voila site_optimizer.ipynb --port 8888
```

---

## Phase 4: Coverage Mosaic (Persona A — Redundanz-Analyse)

### **Konzept: "Welche Region wird von mehreren References abgedeckt?"**

```python
def compute_coverage_mosaic(
    coverage_datasets: dict[str, xr.Dataset]  # {'Ref_A': ds_a, 'Ref_B': ds_b, ...}
) -> xr.Dataset:
    """
    Superpose alle Coverage-Grids. Output zeigt pro Punkt:
    - Beste Klasse (max über alle References)
    - n_references_with_class_2 (Redundanz)
    - Variabilität über References
    """
    
    all_classes = xr.concat(
        [ds['class'] for ds in coverage_datasets.values()],
        dim='reference'
    )
    
    # Best Class: Maximum über alle References (2 ist besser als 1 ist besser als 0)
    best_class = all_classes.max(dim='reference')
    
    # Redundanz: Wie viele References geben Klasse 2?
    n_class_2 = (all_classes == 2).sum(dim='reference')
    
    # Variabilität: Std dev über References
    variability = all_classes.std(dim='reference')
    
    return xr.Dataset({
        'best_class': best_class,
        'n_references_with_class_2': n_class_2,
        'variability': variability,
        'consensus': (variability < 0.5).astype(int)  # Alle References agree?
    })
```

**Visualization:**
```
Layer: Coverage Mosaic
├─ Base Color: best_class
│  ├─ Grün (Klasse 2) — Optimal
│  ├─ Orange (Klasse 1) — Bedingt
│  └─ Rot (Klasse 0) — Nicht abgedeckt
│
├─ Pattern: Hatching bei low consensus
│  ├─ Solid → Alle References agree
│  └─ Hatched → References disagree (Unsicherheit)
│
└─ Info Panel:
   ├─ Click auf Fläche → "2 von 3 References sagen Klasse 2"
   └─ Implikation: "Robust gegen 1 Reference-Ausfall"
```

---

## Zusammenfassung: Implementierungs-Roadmap

| **Phase** | **Modul** | **Persona** | **Aufwand** | **MVP-reif?** |
|-----------|-----------|-----------|----------|--------------|
| **1** | Extended Metadata + QGIS Diagram | Persona B (Analyse) | 4–6h | ✅ JA |
| **2** | Coverage Heatmap (gridded) | Persona A (Planung) | 8–12h | ✅ JA (nach Phase 1) |
| **3** | Site-Placement Optimizer | Persona A (interaktiv) | 4–6h | 🤔 OPTIONAL |
| **4** | Coverage Mosaic | Persona A (Redundanz) | 6–8h | 🤔 OPTIONAL (nach Phase 2) |

---

## Implementierungs-Priorität: Empfehlung

### **Blocker: Persona-Definition**

Frage an Nutzer/Stakeholder:

> *"Gibt es eher eine Person, die Messfeldplanung macht (Persona A), oder jemanden, der die gemessenen Daten analysiert (Persona B)? Oder beides?"*

**Falls Persona B dominiert:**
- **Phase 1 sofort** (2–3 Tage)
- Phase 2/3/4 später (wenn Bedarf wächst)

**Falls Persona A dominiert:**
- **Phase 1 + Phase 2 zusammen** (1 Woche)
- Phase 3/4 später

**Falls beide:**
- **Alle vier Phasen in Serie** (2–3 Wochen)

---

## Konkrete Nächste Schritte

1. **Heute/Morgen:** Personas klären
   - Wer nutzt das Tool? Messfeldplaner? Datenanalyst? Beide?
   - Was ist der primäre Use Case?

2. **Dann:** Phase 1 starten
   - Extended Metadata berechnen
   - GeoPackage erweitern
   - QGIS-Layer + HTML-Tabelle

3. **Parallel:** Rechenzeit validieren
   - Einen Referenz-Punkt × 25k Gitterpunkte durchlaufen
   - Ist 100ms/Punkt realistisch?
   - Parallelisierung mit Dask testen

4. **Dann:** Phase 2 (falls Phase 1 validiert)
   - Coverage Grid implementieren
   - Pro Reference Raster + Polygone

---

## Code-Template Vorschlag

**Neue Datei: `extended_results.py`**
```python
# Berechne PairingMetadata für alle Paarings
# Speichere in ResultSummary.pairings
# Exportiere in GeoPackage + summary.yaml
```

**Neue Datei: `coverage_analyzer.py`** (nur falls Phase 2 gebraucht)
```python
# GriddedRIX-Berechnung
# Coverage Dataset generation
# GeoPackage + Raster Export
```

**QGIS Template:**
```qml
<!-- QGIS Styling für pairings Layer -->
<!-- Farben, Dicke, Label, Popup -->
```

Lass mich wissen, welche Persona relevant ist, und ich starte sofort mit konkretem Code! 🚀
