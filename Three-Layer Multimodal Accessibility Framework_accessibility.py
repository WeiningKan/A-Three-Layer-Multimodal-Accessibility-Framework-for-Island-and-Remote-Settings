# 01_ak_hi_accessibility.py
# ============================================================
# Three-layer multimodal healthcare accessibility algorithm
# for Alaska (AK) and Hawaii (HI)
#
# Layers:
#   Layer 1 — Road-only (ArcGIS OD Cost Matrix)
#   Layer 2 — Road + commercial flight (BTS T-100 data)
#   Layer 3 — Snap routing for AK BGs unreachable by road
#             (Haversine distance to nearest airport → flight)
#
# Prerequisites:
#   - ArcGIS Pro with Network Analyst + Streetmap Premium
#   - ArcGIS geodatabase with: us_bg, aviation_facilities,
#     poi_advan_<NAICS_CODE> layers (see README for data sources)
#
# Output:
#   output/<NAICS_CODE>/parquet/AK_result_<NAICS_CODE>.parquet
#   output/<NAICS_CODE>/parquet/HI_result_<NAICS_CODE>.parquet
#   output/<NAICS_CODE>/parquet/AK_result_all.parquet
#   output/<NAICS_CODE>/parquet/HI_result_all.parquet
#
# Before running:
#   Set WD, WK_GDB, NAICS_GDB, and NDS paths below.
# ============================================================

import arcpy
import arcpy.nax as nax
import pandas as pd
import os
import glob
import math
import numpy as np
from sklearn.linear_model import LinearRegression

arcpy.env.overwriteOutput = True

# ============================================================
# BLOCK 1a: CONFIGURATION
# ============================================================

# --- Directories ---
WD        = r"C:\path\to\your\project"  # <-- replace with your working directory
WK_GDB    = fr"{WD}\nax_proxy_ak_hi\snapping_proxy_WK\snapping_proxy.gdb" # <-- replace with your working geodatabase
NAICS_GDB = fr"{WD}\nax_proxy_ak_hi\gecc_project\gecc_project.gdb" # <-- replace with your NAICS geodatabase
NDS       = r"C:\StreetMapPremium\NA_2025R4_Routing_FGDB\FGDB\StreetMap_Data\NA_Routing.gdb\Routing\Routing_ND"
# NDS: path to your ArcGIS Streetmap Premium network dataset

# --- Input layers ---
US_BG_FC    = fr"{WK_GDB}\us_bg"
AVIATION_FC = fr"{WK_GDB}\aviation_facilities"

# --- Block 3 outputs: BG polygons, centroids, components ---
AK_BG_POLY        = fr"{WK_GDB}\AK_bg"
HI_BG_POLY        = fr"{WK_GDB}\HI_bg"
AK_DISS           = fr"{WK_GDB}\AK_diss"
HI_DISS           = fr"{WK_GDB}\HI_diss"
AK_COMP           = fr"{WK_GDB}\AK_components"
HI_COMP           = fr"{WK_GDB}\HI_components"
AK_BG_PTS         = fr"{WK_GDB}\AK_bg_wtd_pts"
HI_BG_PTS         = fr"{WK_GDB}\HI_bg_wtd_pts"
AK_BG_PTS_COMP    = fr"{WK_GDB}\AK_bg_wtd_pts_with_comp"
HI_BG_PTS_COMP    = fr"{WK_GDB}\HI_bg_wtd_pts_with_comp"

# --- Block 4 outputs: airport point layers ---
AK_AIRPORT_FC     = fr"{WK_GDB}\AK_airports"
HI_AIRPORT_FC     = fr"{WK_GDB}\HI_airports"
AK_AIRPORT_COMP   = fr"{WK_GDB}\AK_airports_with_comp"
HI_AIRPORT_COMP   = fr"{WK_GDB}\HI_airports_with_comp"
AK_OOS_AIRPORT_FC = fr"{WK_GDB}\AK_oos_airports"
AK_FULL_AIRPORT_FC = fr"{WK_GDB}\AK_full_airports"
HI_FULL_AIRPORT_FC = fr"{WK_GDB}\HI_full_airports"

# --- Field names ---
ORIG_ID_NAME = "GEOID_bg"
DEST_ID_NAME = "PLACEKEY"

# --- Data files ---
ENPLANEMENT = fr"{WD}\data\arp_cy2024_commercial_service_enplanements.xlsx"
SEGMENT     = fr"{WD}\data\T_T100D_SEGMENT_US_CARRIER_ONLY.csv"
FLIGHT_TIME = fr"{WD}\data\flight_time.xlsx"

# --- Parameters ---
IMPEDANCE_CUTOFF     = 180   # minutes, max total travel time
AIRPORT_DRIVE_CUTOFF = 60    # minutes, max drive time BG -> departure airport
BUFFER_DIST          = 1000  # miles, candidate POI search radius

# --- NAICS codes to process ---
NAICS_CODES = [
    "446110", "621320", "622310", "621340",
    "621492", "621498", "621511", "621512",
    "622110", "621493", "812191", "621420", "621111"
]

# --- Output ---
OUT_DIR     = fr"{WD}\output\621420"   # base output dir (parquet goes inside)
PARQUET_DIR = fr"{OUT_DIR}\parquet"
os.makedirs(PARQUET_DIR, exist_ok=True)


# ============================================================
# BLOCK 1b: HELPER FUNCTIONS
# ============================================================

def run_odcm(orig_fc, dest_fc, cutoff,
             orig_name_field=ORIG_ID_NAME, dest_name_field=DEST_ID_NAME):
    """Run ArcGIS OD Cost Matrix and return the result object."""
    odcm = nax.OriginDestinationCostMatrix(NDS)
    travel_modes = nax.GetTravelModes(NDS)
    mode = travel_modes["Driving Time"]
    mode.useHierarchy = "USE_HIERARCHY"
    odcm.travelMode = mode
    odcm.timeUnits = nax.TimeUnits.Minutes
    odcm.lineShapeType = nax.LineShapeType.NoLine
    odcm.defaultImpedanceCutoff = cutoff

    ofm = odcm.fieldMappings(nax.OriginDestinationCostMatrixInputDataType.Origins)
    ofm["Name"].mappedFieldName = orig_name_field
    dfm = odcm.fieldMappings(nax.OriginDestinationCostMatrixInputDataType.Destinations)
    dfm["Name"].mappedFieldName = dest_name_field

    odcm.load(nax.OriginDestinationCostMatrixInputDataType.Origins, orig_fc, ofm)
    odcm.load(nax.OriginDestinationCostMatrixInputDataType.Destinations, dest_fc, dfm)

    return odcm.solve()


def get_status(travel_time):
    """Classify travel time into status string."""
    if travel_time is None:
        return "unreachable"
    elif travel_time <= 180:
        return "reachable"
    elif travel_time <= 240:
        return "gt_3h"
    elif travel_time <= 360:
        return "gt_4h"
    else:
        return "gt_6h"


def haversine_miles(xy1, xy2):
    """Compute great-circle distance in miles between two (lon, lat) points."""
    lon1, lat1 = math.radians(xy1[0]), math.radians(xy1[1])
    lon2, lat2 = math.radians(xy2[0]), math.radians(xy2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * 3958.8 * math.asin(math.sqrt(a))


def count_null_component(fc, field_name="component_id"):
    total, null = 0, 0
    with arcpy.da.SearchCursor(fc, [field_name]) as cursor:
        for row in cursor:
            total += 1
            if row[0] is None:
                null += 1
    return total, null


print("Block 1 done: configuration and helpers loaded.")


# ============================================================
# BLOCK 2a: GENERATE FLIGHT TIME LOOKUP TABLE (BTS T-100)
# ============================================================

df_segment = pd.read_csv(SEGMENT, usecols=[
    "DEPARTURES_PERFORMED", "RAMP_TO_RAMP", "DISTANCE",
    "ORIGIN", "ORIGIN_STATE_ABR",
    "DEST", "DEST_STATE_ABR",
    "CLASS"
])

df_segment = df_segment[df_segment["CLASS"] == "F"]
df_segment = df_segment[df_segment["ORIGIN"] != df_segment["DEST"]]
df_segment = df_segment[df_segment["DEPARTURES_PERFORMED"] > 0]
df_segment = df_segment[df_segment["RAMP_TO_RAMP"] > 0]

df_flight_time = (
    df_segment
    .groupby(["ORIGIN", "ORIGIN_STATE_ABR", "DEST", "DEST_STATE_ABR", "DISTANCE"])
    .apply(lambda x: x["RAMP_TO_RAMP"].sum() / x["DEPARTURES_PERFORMED"].sum(),
           include_groups=False)
    .reset_index(name="flight_time_min")
)

df_flight_time.to_excel(FLIGHT_TIME, index=False)
print(f"Block 2 done: {len(df_flight_time)} OD pairs saved to flight_time.xlsx")


# ============================================================
# BLOCK 2b: OLS INTERPOLATION FOR MISSING AK AIRPORT PAIRS
# ============================================================

df_seg = pd.read_csv(SEGMENT, usecols=[
    "DEPARTURES_PERFORMED", "RAMP_TO_RAMP",
    "ORIGIN", "ORIGIN_STATE_ABR",
    "DEST", "DEST_STATE_ABR",
    "DISTANCE", "CLASS"
])

df_ak_seg = df_seg[
    (df_seg["CLASS"] == "F") &
    (df_seg["DEPARTURES_PERFORMED"] > 0) &
    (df_seg["RAMP_TO_RAMP"] > 0) &
    (df_seg["DISTANCE"] > 0) &
    (
        (df_seg["ORIGIN_STATE_ABR"] == "AK") |
        (df_seg["DEST_STATE_ABR"] == "AK")
    )
].copy()

df_ak_model = (
    df_ak_seg
    .groupby(["ORIGIN", "DEST", "DISTANCE"])
    .apply(lambda x: x["RAMP_TO_RAMP"].sum() / x["DEPARTURES_PERFORMED"].sum(),
           include_groups=False)
    .reset_index(name="flight_time_min")
)

X = df_ak_model[["DISTANCE"]].values
y = df_ak_model["flight_time_min"].values
model = LinearRegression()
model.fit(X, y)
r2 = model.score(X, y)
print(f"OLS model: flight_time = {model.coef_[0]:.4f} × distance + {model.intercept_:.4f}")
print(f"R²: {r2:.4f}, training data: {len(df_ak_model)} OD pairs")

t100_ak_airports = set(df_ak_seg["ORIGIN"].tolist() + df_ak_seg["DEST"].tolist())
df_enplan = pd.read_excel(ENPLANEMENT)
df_enplan_ak = df_enplan[df_enplan["ST"] == "AK"].copy()
enplan_ak_airports = set(df_enplan_ak["Locid"].tolist())
missing_airports = enplan_ak_airports - t100_ak_airports
print(f"AK airports in enplanement but missing from T-100: {len(missing_airports)}")

all_airport_coords = {}
with arcpy.da.SearchCursor(AK_AIRPORT_COMP, ["Locid", "SHAPE@XY"]) as cursor:
    for row in cursor:
        all_airport_coords[row[0]] = row[1]

missing_airport_coords = {
    loc: all_airport_coords[loc]
    for loc in missing_airports if loc in all_airport_coords
}
enplan_dest_airports = {
    loc: all_airport_coords[loc]
    for loc in enplan_ak_airports if loc in all_airport_coords
}

new_rows = []
for origin_loc, origin_xy in missing_airport_coords.items():
    for dest_loc, dest_xy in enplan_dest_airports.items():
        if origin_loc == dest_loc:
            continue
        existing = df_flight_time[
            (df_flight_time["ORIGIN"] == origin_loc) &
            (df_flight_time["DEST"] == dest_loc)
        ]
        if len(existing) > 0:
            continue
        dist = haversine_miles(origin_xy, dest_xy)
        if dist > 500:
            continue
        predicted_time = max(model.predict([[dist]])[0], 5.0)
        new_rows.append({
            "ORIGIN": origin_loc, "ORIGIN_STATE_ABR": "AK",
            "DEST": dest_loc,    "DEST_STATE_ABR": "AK",
            "flight_time_min": round(predicted_time, 2)
        })

if new_rows:
    df_new = pd.DataFrame(new_rows)
    df_flight_time = pd.concat([df_flight_time, df_new], ignore_index=True)
    df_flight_time.to_excel(FLIGHT_TIME, index=False)
    print(f"Block 2b done: {len(df_new)} new OD pairs added, total {len(df_flight_time)}")
else:
    print("Block 2b done: no new OD pairs to add.")


# ============================================================
# BLOCK 3: BG POLYGONS, COMPONENTS, CENTROIDS, SPATIAL JOIN
# ============================================================

# 3a: Extract AK and HI BG polygons
arcpy.analysis.Select(US_BG_FC, AK_BG_POLY, "STATE_FIPS = '02'")
arcpy.analysis.Select(US_BG_FC, HI_BG_POLY, "STATE_FIPS = '15'")
print(f"AK BG polygons: {int(arcpy.management.GetCount(AK_BG_POLY)[0])}")
print(f"HI BG polygons: {int(arcpy.management.GetCount(HI_BG_POLY)[0])}")

# 3b: Dissolve
arcpy.management.Dissolve(AK_BG_POLY, AK_DISS)
arcpy.management.Dissolve(HI_BG_POLY, HI_DISS)

# 3c: Multipart to singlepart
arcpy.management.MultipartToSinglepart(AK_DISS, AK_COMP)
arcpy.management.MultipartToSinglepart(HI_DISS, HI_COMP)
print(f"AK components: {int(arcpy.management.GetCount(AK_COMP)[0])}")
print(f"HI components: {int(arcpy.management.GetCount(HI_COMP)[0])}")

# 3d: Add component_id
def add_component_id(fc, field_name="component_id"):
    if field_name not in [f.name for f in arcpy.ListFields(fc)]:
        arcpy.management.AddField(fc, field_name, "LONG")
    with arcpy.da.UpdateCursor(fc, ["OID@", field_name]) as cursor:
        for row in cursor:
            row[1] = row[0]
            cursor.updateRow(row)

add_component_id(AK_COMP)
add_component_id(HI_COMP)

# 3e: Spatial join: BG centroids -> components
arcpy.analysis.SpatialJoin(
    target_features=AK_BG_PTS, join_features=AK_COMP,
    out_feature_class=AK_BG_PTS_COMP,
    join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL",
    match_option="INTERSECT"
)
arcpy.analysis.SpatialJoin(
    target_features=HI_BG_PTS, join_features=HI_COMP,
    out_feature_class=HI_BG_PTS_COMP,
    join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL",
    match_option="INTERSECT"
)

ak_total, ak_null = count_null_component(AK_BG_PTS_COMP)
hi_total, hi_null = count_null_component(HI_BG_PTS_COMP)
print(f"AK: {ak_total} points, {ak_null} null component_id")
print(f"HI: {hi_total} points, {hi_null} null component_id")

# Load BG data into memory
ak_bg_data = {}
with arcpy.da.SearchCursor(AK_BG_PTS_COMP, ["GEOID_bg", "component_id", "SHAPE@XY"]) as cursor:
    for row in cursor:
        ak_bg_data[str(row[0]).zfill(12)] = {"component_id": row[1], "xy": row[2]}

hi_bg_data = {}
with arcpy.da.SearchCursor(HI_BG_PTS_COMP, ["GEOID_bg", "component_id", "SHAPE@XY"]) as cursor:
    for row in cursor:
        hi_bg_data[str(row[0]).zfill(12)] = {"component_id": row[1], "xy": row[2]}

print(f"Block 3 done: AK={len(ak_bg_data)} BGs, HI={len(hi_bg_data)} BGs")


# ============================================================
# BLOCK 4: BUILD AIRPORT POINT LAYERS
# ============================================================

df_enplane = pd.read_excel(ENPLANEMENT, usecols=["Locid", "ST", "S/L", "CY 24 Enplanements"])
df_enplane_ak_hi = df_enplane[df_enplane["ST"].isin(["AK", "HI"])].copy()

avfac_data = {row[0]: row[1] for row in arcpy.da.SearchCursor(AVIATION_FC, ["ARPT_ID", "SHAPE@XY"])}
df_enplane_ak_hi["geometry"] = df_enplane_ak_hi["Locid"].map(avfac_data)
df_airports = df_enplane_ak_hi[df_enplane_ak_hi["geometry"].notna()].copy()
print(f"Matched airports: {len(df_airports)}")

def create_airport_fc(df, state, out_fc):
    df_state = df[df["ST"] == state].copy()
    sr = arcpy.Describe(AVIATION_FC).spatialReference
    arcpy.management.CreateFeatureclass(
        out_path=WK_GDB,
        out_name=out_fc.split("\\")[-1],
        geometry_type="POINT",
        spatial_reference=sr
    )
    arcpy.management.AddField(out_fc, "Locid", "TEXT", field_length=10)
    arcpy.management.AddField(out_fc, "ST", "TEXT", field_length=2)
    arcpy.management.AddField(out_fc, "SL", "TEXT", field_length=2)
    arcpy.management.AddField(out_fc, "CY24_Enplanements", "LONG")
    fields = ["SHAPE@XY", "Locid", "ST", "SL", "CY24_Enplanements"]
    with arcpy.da.InsertCursor(out_fc, fields) as cursor:
        for _, row in df_state.iterrows():
            cursor.insertRow([
                row["geometry"], row["Locid"], row["ST"],
                row["S/L"], int(row["CY 24 Enplanements"])
            ])
    print(f"{state} airports created: {int(arcpy.management.GetCount(out_fc)[0])}")

create_airport_fc(df_airports, "AK", AK_AIRPORT_FC)
create_airport_fc(df_airports, "HI", HI_AIRPORT_FC)

arcpy.analysis.SpatialJoin(
    target_features=AK_AIRPORT_FC, join_features=AK_COMP,
    out_feature_class=AK_AIRPORT_COMP,
    join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL",
    match_option="INTERSECT"
)
arcpy.analysis.SpatialJoin(
    target_features=HI_AIRPORT_FC, join_features=HI_COMP,
    out_feature_class=HI_AIRPORT_COMP,
    join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL",
    match_option="INTERSECT"
)
print("Block 4 done.")


# ============================================================
# BLOCK 5: BATCH RUN ALL NAICS CODES — AK & HI
# ============================================================

ak_all_results = []
hi_all_results = []

for NAICS_CODE in NAICS_CODES:
    print(f"\n{'='*60}")
    print(f"Processing NAICS: {NAICS_CODE}")
    print(f"{'='*60}")

    POI_FC = fr"{NAICS_GDB}\poi_advan_{NAICS_CODE}"
    if not arcpy.Exists(POI_FC):
        print(f"  POI layer not found: {POI_FC}, skipping.")
        continue

    AK_OUT_DIR = fr"{WD}\output\{NAICS_CODE}\AK"
    HI_OUT_DIR = fr"{WD}\output\{NAICS_CODE}\HI"
    os.makedirs(AK_OUT_DIR, exist_ok=True)
    os.makedirs(HI_OUT_DIR, exist_ok=True)

    # --- Load POI data ---
    poi_data = {}
    with arcpy.da.SearchCursor(POI_FC, ["PLACEKEY", "SHAPE@XY"]) as cursor:
        for row in cursor:
            poi_data[row[0]] = row[1]

    # --- Build candidate POI sets (within BUFFER_DIST miles) ---
    ak_candidates = {}
    for bg_id, info in ak_bg_data.items():
        bg_xy = info["xy"]
        ak_candidates[bg_id] = [
            placekey for placekey, poi_xy in poi_data.items()
            if haversine_miles(bg_xy, poi_xy) <= BUFFER_DIST
        ]

    hi_candidates = {}
    for bg_id, info in hi_bg_data.items():
        bg_xy = info["xy"]
        hi_candidates[bg_id] = [
            placekey for placekey, poi_xy in poi_data.items()
            if haversine_miles(bg_xy, poi_xy) <= BUFFER_DIST
        ]

    # --------------------------------------------------------
    # AK: Travel time calculation (Layer 1 + 2 + 3)
    # --------------------------------------------------------
    print(f"[{NAICS_CODE}] AK: travel time calculation...")

    for f in glob.glob(fr"{AK_OUT_DIR}\comp_*.csv"):
        os.remove(f)

    ak_airport_data = {}
    with arcpy.da.SearchCursor(AK_FULL_AIRPORT_FC, ["Locid", "component_id", "SHAPE@XY"]) as cursor:
        for row in cursor:
            ak_airport_data[row[0]] = {"component_id": row[1], "xy": row[2]}

    ak_departure_airports = {
        locid for locid, info in ak_airport_data.items()
        if info["component_id"] is not None
    }
    ak_flight_lookup = {
        (row["ORIGIN"], row["DEST"]): row["flight_time_min"]
        for _, row in df_flight_time[
            df_flight_time["ORIGIN"].isin(ak_departure_airports)
        ].iterrows()
    }
    ak_instate_airport_coords = {}
    with arcpy.da.SearchCursor(AK_AIRPORT_COMP, ["Locid", "SHAPE@XY"]) as cursor:
        for row in cursor:
            ak_instate_airport_coords[row[0]] = row[1]

    ak_airports_with_departures = {dep for dep, dest in ak_flight_lookup.keys()}
    ak_components = sorted({
        info["component_id"] for info in ak_bg_data.values()
        if info["component_id"] is not None
    })

    for comp_id in ak_components:
        try:
            bg_ids_in_comp = [
                bg_id for bg_id, info in ak_bg_data.items()
                if info["component_id"] == comp_id
            ]
            candidate_pois = set()
            for bg_id in bg_ids_in_comp:
                candidate_pois.update(ak_candidates[bg_id])
            if not candidate_pois:
                continue

            poi_where = f"PLACEKEY IN ({','.join([chr(39)+p+chr(39) for p in candidate_pois])})"

            # Layer 1: Road-only
            orig_lyr = arcpy.management.MakeFeatureLayer(AK_BG_PTS_COMP, "orig_lyr").getOutput(0)
            arcpy.management.SelectLayerByAttribute(orig_lyr, "NEW_SELECTION", f"component_id = {comp_id}")
            orig_sel = arcpy.management.CopyFeatures(orig_lyr, "in_memory/orig_sel").getOutput(0)
            dest_lyr = arcpy.management.MakeFeatureLayer(POI_FC, "dest_lyr").getOutput(0)
            arcpy.management.SelectLayerByAttribute(dest_lyr, "NEW_SELECTION", poi_where)
            dest_sel = arcpy.management.CopyFeatures(dest_lyr, "in_memory/dest_sel").getOutput(0)

            result_road = run_odcm(orig_sel, dest_sel, IMPEDANCE_CUTOFF)
            road_results = {}
            if result_road.solveSucceeded:
                with result_road.searchCursor(
                    nax.OriginDestinationCostMatrixOutputDataType.Lines,
                    ["OriginName", "DestinationName", "Total_Time"]) as cur:
                    for row in cur:
                        road_results[(row[0], row[1])] = row[2]

            arcpy.management.Delete(orig_sel)
            arcpy.management.Delete(dest_sel)
            arcpy.management.Delete(orig_lyr)
            arcpy.management.Delete(dest_lyr)

            # Layer 2: Multimodal (road + flight)
            comp_airports = {
                locid: info for locid, info in ak_airport_data.items()
                if info["component_id"] == comp_id
            }
            bg_to_airport = {}
            airport_to_poi = {}
            all_dest_airports = set()

            if comp_airports:
                orig_lyr2 = arcpy.management.MakeFeatureLayer(AK_BG_PTS_COMP, "orig_lyr2").getOutput(0)
                arcpy.management.SelectLayerByAttribute(orig_lyr2, "NEW_SELECTION", f"component_id = {comp_id}")
                orig_sel2 = arcpy.management.CopyFeatures(orig_lyr2, "in_memory/orig_sel2").getOutput(0)
                airport_where = f"Locid IN ({','.join([chr(39)+l+chr(39) for l in comp_airports.keys()])})"
                airport_lyr = arcpy.management.MakeFeatureLayer(AK_FULL_AIRPORT_FC, "airport_lyr").getOutput(0)
                arcpy.management.SelectLayerByAttribute(airport_lyr, "NEW_SELECTION", airport_where)
                airport_sel = arcpy.management.CopyFeatures(airport_lyr, "in_memory/airport_sel").getOutput(0)

                result_bg_airport = run_odcm(orig_sel2, airport_sel, AIRPORT_DRIVE_CUTOFF, dest_name_field="Locid")
                if result_bg_airport.solveSucceeded:
                    with result_bg_airport.searchCursor(
                        nax.OriginDestinationCostMatrixOutputDataType.Lines,
                        ["OriginName", "DestinationName", "Total_Time"]) as cur:
                        for row in cur:
                            bg_to_airport[(row[0], row[1])] = row[2]

                arcpy.management.Delete(orig_sel2)
                arcpy.management.Delete(airport_sel)
                arcpy.management.Delete(orig_lyr2)
                arcpy.management.Delete(airport_lyr)

                all_dest_airports = {
                    dest for dep, dest in ak_flight_lookup.keys()
                    if dep in comp_airports
                }

                if all_dest_airports:
                    dest_airport_where = f"Locid IN ({','.join([chr(39)+l+chr(39) for l in all_dest_airports])})"
                    dest_airport_lyr = arcpy.management.MakeFeatureLayer(AK_FULL_AIRPORT_FC, "dest_airport_lyr").getOutput(0)
                    arcpy.management.SelectLayerByAttribute(dest_airport_lyr, "NEW_SELECTION", dest_airport_where)
                    dest_airport_sel = arcpy.management.CopyFeatures(dest_airport_lyr, "in_memory/dest_airport_sel").getOutput(0)
                    dest_lyr2 = arcpy.management.MakeFeatureLayer(POI_FC, "dest_lyr2").getOutput(0)
                    arcpy.management.SelectLayerByAttribute(dest_lyr2, "NEW_SELECTION", poi_where)
                    dest_sel2 = arcpy.management.CopyFeatures(dest_lyr2, "in_memory/dest_sel2").getOutput(0)

                    result_airport_poi = run_odcm(dest_airport_sel, dest_sel2, IMPEDANCE_CUTOFF, orig_name_field="Locid")
                    if result_airport_poi.solveSucceeded:
                        with result_airport_poi.searchCursor(
                            nax.OriginDestinationCostMatrixOutputDataType.Lines,
                            ["OriginName", "DestinationName", "Total_Time"]) as cur:
                            for row in cur:
                                airport_to_poi[(row[0], row[1])] = row[2]

                    arcpy.management.Delete(dest_airport_sel)
                    arcpy.management.Delete(dest_sel2)
                    arcpy.management.Delete(dest_airport_lyr)
                    arcpy.management.Delete(dest_lyr2)

            # Compile Layer 1 + Layer 2
            comp_rows = []
            for bg_id in bg_ids_in_comp:
                for placekey in ak_candidates[bg_id]:
                    if (bg_id, placekey) in road_results:
                        t = road_results[(bg_id, placekey)]
                        comp_rows.append({
                            "OriginName": bg_id, "DestinationName": placekey,
                            "travel_time_min": t, "status": get_status(t), "method": "road"
                        })
                        continue
                    best_time = None
                    for dep_locid in comp_airports.keys():
                        drive_to_dep = bg_to_airport.get((bg_id, dep_locid))
                        if drive_to_dep is None:
                            continue
                        for arr_locid in all_dest_airports:
                            flight_time = ak_flight_lookup.get((dep_locid, arr_locid))
                            if flight_time is None:
                                continue
                            drive_from_arr = airport_to_poi.get((arr_locid, placekey))
                            if drive_from_arr is None:
                                continue
                            total = drive_to_dep + flight_time + drive_from_arr
                            if best_time is None or total < best_time:
                                best_time = total
                    comp_rows.append({
                        "OriginName": bg_id, "DestinationName": placekey,
                        "travel_time_min": best_time, "status": get_status(best_time),
                        "method": "multimodal"
                    })

            # Layer 3: Snap routing (AK only)
            # Snap time formula: t_snap = (haversine_distance_miles / 70) * 60
            solved_in_l1_l2 = {
                row["OriginName"] for row in comp_rows
                if row["status"] != "unreachable"
            }
            unreachable_bgs = [
                bg_id for bg_id in bg_ids_in_comp
                if bg_id not in solved_in_l1_l2
            ]

            if unreachable_bgs and ak_airports_with_departures:
                unique_dest_airports = set()
                bg_snap_info = {}

                for bg_id in unreachable_bgs:
                    bg_xy = ak_bg_data[bg_id]["xy"]
                    best_dist, best_airport = float("inf"), None
                    for locid, xy in ak_instate_airport_coords.items():
                        if locid not in ak_airports_with_departures:
                            continue
                        dist = haversine_miles(bg_xy, xy)
                        if dist < best_dist:
                            best_dist, best_airport = dist, locid

                    if best_airport is None:
                        continue

                    snap_time = (best_dist / 70) * 60   # estimated overland travel time
                    dest_airports_for_snap = [
                        dest for (dep, dest) in ak_flight_lookup.keys()
                        if dep == best_airport
                    ]
                    unique_dest_airports.update(dest_airports_for_snap)
                    bg_snap_info[bg_id] = {
                        "snap_airport": best_airport,
                        "snap_time": snap_time,
                        "dest_airports": dest_airports_for_snap
                    }

                snap_airport_to_poi = {}
                if unique_dest_airports:
                    snap_candidate_pois = set()
                    for bg_id in unreachable_bgs:
                        snap_candidate_pois.update(ak_candidates[bg_id])

                    if snap_candidate_pois:
                        snap_poi_where = f"PLACEKEY IN ({','.join([chr(39)+p+chr(39) for p in snap_candidate_pois])})"
                        snap_dest_where = f"Locid IN ({','.join([chr(39)+l+chr(39) for l in unique_dest_airports])})"

                        snap_dest_lyr = arcpy.management.MakeFeatureLayer(AK_FULL_AIRPORT_FC, "snap_dest_lyr").getOutput(0)
                        arcpy.management.SelectLayerByAttribute(snap_dest_lyr, "NEW_SELECTION", snap_dest_where)
                        snap_dest_sel = arcpy.management.CopyFeatures(snap_dest_lyr, "in_memory/snap_dest_sel").getOutput(0)
                        snap_poi_lyr = arcpy.management.MakeFeatureLayer(POI_FC, "snap_poi_lyr").getOutput(0)
                        arcpy.management.SelectLayerByAttribute(snap_poi_lyr, "NEW_SELECTION", snap_poi_where)
                        snap_poi_sel = arcpy.management.CopyFeatures(snap_poi_lyr, "in_memory/snap_poi_sel").getOutput(0)

                        result_snap = run_odcm(snap_dest_sel, snap_poi_sel, IMPEDANCE_CUTOFF, orig_name_field="Locid")
                        if result_snap.solveSucceeded:
                            with result_snap.searchCursor(
                                nax.OriginDestinationCostMatrixOutputDataType.Lines,
                                ["OriginName", "DestinationName", "Total_Time"]) as cur:
                                for row in cur:
                                    snap_airport_to_poi[(row[0], row[1])] = row[2]

                        arcpy.management.Delete(snap_dest_sel)
                        arcpy.management.Delete(snap_poi_sel)
                        arcpy.management.Delete(snap_dest_lyr)
                        arcpy.management.Delete(snap_poi_lyr)

                for bg_id in unreachable_bgs:
                    if bg_id not in bg_snap_info:
                        for placekey in ak_candidates[bg_id]:
                            comp_rows.append({
                                "OriginName": bg_id, "DestinationName": placekey,
                                "travel_time_min": None, "status": "unreachable", "method": "snap"
                            })
                        continue

                    snap_time = bg_snap_info[bg_id]["snap_time"]
                    for placekey in ak_candidates[bg_id]:
                        best_time = None
                        for arr_locid in bg_snap_info[bg_id]["dest_airports"]:
                            flight_time = ak_flight_lookup.get((bg_snap_info[bg_id]["snap_airport"], arr_locid))
                            if flight_time is None:
                                continue
                            drive_from_arr = snap_airport_to_poi.get((arr_locid, placekey))
                            if drive_from_arr is None:
                                continue
                            total = snap_time + flight_time + drive_from_arr
                            if best_time is None or total < best_time:
                                best_time = total
                        comp_rows.append({
                            "OriginName": bg_id, "DestinationName": placekey,
                            "travel_time_min": best_time, "status": get_status(best_time),
                            "method": "snap"
                        })

            df_comp = pd.DataFrame(comp_rows)
            df_comp["OriginName"] = df_comp["OriginName"].astype(str).str.zfill(12)
            df_comp.to_csv(fr"{AK_OUT_DIR}\comp_{comp_id}.csv", index=False)

        except Exception as e:
            print(f"ERROR AK component {comp_id}: {e}")

    # AK: Summarize
    all_files = glob.glob(fr"{AK_OUT_DIR}\comp_*.csv")
    df_ak_all = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    df_ak_all["OriginName"] = df_ak_all["OriginName"].astype(str).str.zfill(12)
    df_ak_all["layer"] = df_ak_all["method"].map({"road": 1, "multimodal": 2, "snap": 3})

    df_ak_reachable = df_ak_all[df_ak_all["status"] != "unreachable"].copy()
    df_ak_min = (
        df_ak_reachable
        .sort_values("travel_time_min")
        .groupby("OriginName").first()
        .reset_index()
        [["OriginName", "DestinationName", "travel_time_min", "status", "method", "layer"]]
    )
    solved_bg_ids = set(df_ak_min["OriginName"])
    unreachable_bg_ids = [
        bg_id for bg_id in df_ak_all["OriginName"].unique()
        if bg_id not in solved_bg_ids
    ]
    if unreachable_bg_ids:
        df_ak_result = pd.concat([
            df_ak_min,
            pd.DataFrame({
                "OriginName": unreachable_bg_ids,
                "DestinationName": pd.NA, "travel_time_min": pd.NA,
                "status": "unreachable", "method": pd.NA, "layer": pd.NA
            })
        ], ignore_index=True)
    else:
        df_ak_result = df_ak_min.copy()

    df_ak_result["NAICS_CODE"] = NAICS_CODE
    df_ak_result["OriginName"] = df_ak_result["OriginName"].astype(str).str.zfill(12)
    df_ak_result.to_parquet(fr"{PARQUET_DIR}\AK_result_{NAICS_CODE}.parquet", index=False)
    ak_all_results.append(df_ak_result)
    print(f"  AK done: {len(df_ak_result)} BGs | {df_ak_result['status'].value_counts().to_dict()}")

    # --------------------------------------------------------
    # HI: Travel time calculation (Layer 1 + 2 only, no snap)
    # --------------------------------------------------------
    print(f"[{NAICS_CODE}] HI: travel time calculation...")

    for f in glob.glob(fr"{HI_OUT_DIR}\comp_*.csv"):
        os.remove(f)

    hi_airport_data = {}
    with arcpy.da.SearchCursor(HI_FULL_AIRPORT_FC, ["Locid", "component_id", "SHAPE@XY"]) as cursor:
        for row in cursor:
            hi_airport_data[row[0]] = {"component_id": row[1], "xy": row[2]}

    hi_departure_airports = {
        locid for locid, info in hi_airport_data.items()
        if info["component_id"] is not None
    }
    hi_flight_lookup = {
        (row["ORIGIN"], row["DEST"]): row["flight_time_min"]
        for _, row in df_flight_time[
            df_flight_time["ORIGIN"].isin(hi_departure_airports)
        ].iterrows()
    }

    hi_components = sorted({
        info["component_id"] for info in hi_bg_data.values()
        if info["component_id"] is not None
    })

    for comp_id in hi_components:
        try:
            bg_ids_in_comp = [
                bg_id for bg_id, info in hi_bg_data.items()
                if info["component_id"] == comp_id
            ]
            candidate_pois = set()
            for bg_id in bg_ids_in_comp:
                candidate_pois.update(hi_candidates[bg_id])
            if not candidate_pois:
                continue

            poi_where = f"PLACEKEY IN ({','.join([chr(39)+p+chr(39) for p in candidate_pois])})"

            # Layer 1: Road-only
            orig_lyr = arcpy.management.MakeFeatureLayer(HI_BG_PTS_COMP, "orig_lyr").getOutput(0)
            arcpy.management.SelectLayerByAttribute(orig_lyr, "NEW_SELECTION", f"component_id = {comp_id}")
            orig_sel = arcpy.management.CopyFeatures(orig_lyr, "in_memory/orig_sel").getOutput(0)
            dest_lyr = arcpy.management.MakeFeatureLayer(POI_FC, "dest_lyr").getOutput(0)
            arcpy.management.SelectLayerByAttribute(dest_lyr, "NEW_SELECTION", poi_where)
            dest_sel = arcpy.management.CopyFeatures(dest_lyr, "in_memory/dest_sel").getOutput(0)

            result_road = run_odcm(orig_sel, dest_sel, IMPEDANCE_CUTOFF)
            road_results = {}
            if result_road.solveSucceeded:
                with result_road.searchCursor(
                    nax.OriginDestinationCostMatrixOutputDataType.Lines,
                    ["OriginName", "DestinationName", "Total_Time"]) as cur:
                    for row in cur:
                        road_results[(row[0], row[1])] = row[2]

            arcpy.management.Delete(orig_sel)
            arcpy.management.Delete(dest_sel)
            arcpy.management.Delete(orig_lyr)
            arcpy.management.Delete(dest_lyr)

            # Layer 2: Multimodal
            comp_airports = {
                locid: info for locid, info in hi_airport_data.items()
                if info["component_id"] == comp_id
            }
            bg_to_airport = {}
            airport_to_poi = {}
            all_dest_airports = set()

            if comp_airports:
                orig_lyr2 = arcpy.management.MakeFeatureLayer(HI_BG_PTS_COMP, "orig_lyr2").getOutput(0)
                arcpy.management.SelectLayerByAttribute(orig_lyr2, "NEW_SELECTION", f"component_id = {comp_id}")
                orig_sel2 = arcpy.management.CopyFeatures(orig_lyr2, "in_memory/orig_sel2").getOutput(0)
                airport_where = f"Locid IN ({','.join([chr(39)+l+chr(39) for l in comp_airports.keys()])})"
                airport_lyr = arcpy.management.MakeFeatureLayer(HI_FULL_AIRPORT_FC, "airport_lyr").getOutput(0)
                arcpy.management.SelectLayerByAttribute(airport_lyr, "NEW_SELECTION", airport_where)
                airport_sel = arcpy.management.CopyFeatures(airport_lyr, "in_memory/airport_sel").getOutput(0)

                result_bg_airport = run_odcm(orig_sel2, airport_sel, AIRPORT_DRIVE_CUTOFF, dest_name_field="Locid")
                if result_bg_airport.solveSucceeded:
                    with result_bg_airport.searchCursor(
                        nax.OriginDestinationCostMatrixOutputDataType.Lines,
                        ["OriginName", "DestinationName", "Total_Time"]) as cur:
                        for row in cur:
                            bg_to_airport[(row[0], row[1])] = row[2]

                arcpy.management.Delete(orig_sel2)
                arcpy.management.Delete(airport_sel)
                arcpy.management.Delete(orig_lyr2)
                arcpy.management.Delete(airport_lyr)

                all_dest_airports = {
                    dest for dep, dest in hi_flight_lookup.keys()
                    if dep in comp_airports
                }

                if all_dest_airports:
                    dest_airport_where = f"Locid IN ({','.join([chr(39)+l+chr(39) for l in all_dest_airports])})"
                    dest_airport_lyr = arcpy.management.MakeFeatureLayer(HI_FULL_AIRPORT_FC, "dest_airport_lyr").getOutput(0)
                    arcpy.management.SelectLayerByAttribute(dest_airport_lyr, "NEW_SELECTION", dest_airport_where)
                    dest_airport_sel = arcpy.management.CopyFeatures(dest_airport_lyr, "in_memory/dest_airport_sel").getOutput(0)
                    dest_lyr2 = arcpy.management.MakeFeatureLayer(POI_FC, "dest_lyr2").getOutput(0)
                    arcpy.management.SelectLayerByAttribute(dest_lyr2, "NEW_SELECTION", poi_where)
                    dest_sel2 = arcpy.management.CopyFeatures(dest_lyr2, "in_memory/dest_sel2").getOutput(0)

                    result_airport_poi = run_odcm(dest_airport_sel, dest_sel2, IMPEDANCE_CUTOFF, orig_name_field="Locid")
                    if result_airport_poi.solveSucceeded:
                        with result_airport_poi.searchCursor(
                            nax.OriginDestinationCostMatrixOutputDataType.Lines,
                            ["OriginName", "DestinationName", "Total_Time"]) as cur:
                            for row in cur:
                                airport_to_poi[(row[0], row[1])] = row[2]

                    arcpy.management.Delete(dest_airport_sel)
                    arcpy.management.Delete(dest_sel2)
                    arcpy.management.Delete(dest_airport_lyr)
                    arcpy.management.Delete(dest_lyr2)

            comp_rows = []
            for bg_id in bg_ids_in_comp:
                for placekey in hi_candidates[bg_id]:
                    if (bg_id, placekey) in road_results:
                        t = road_results[(bg_id, placekey)]
                        comp_rows.append({
                            "OriginName": bg_id, "DestinationName": placekey,
                            "travel_time_min": t, "status": get_status(t), "method": "road"
                        })
                        continue
                    best_time = None
                    for dep_locid in comp_airports.keys():
                        drive_to_dep = bg_to_airport.get((bg_id, dep_locid))
                        if drive_to_dep is None:
                            continue
                        for arr_locid in all_dest_airports:
                            flight_time = hi_flight_lookup.get((dep_locid, arr_locid))
                            if flight_time is None:
                                continue
                            drive_from_arr = airport_to_poi.get((arr_locid, placekey))
                            if drive_from_arr is None:
                                continue
                            total = drive_to_dep + flight_time + drive_from_arr
                            if best_time is None or total < best_time:
                                best_time = total
                    comp_rows.append({
                        "OriginName": bg_id, "DestinationName": placekey,
                        "travel_time_min": best_time, "status": get_status(best_time),
                        "method": "multimodal"
                    })

            df_comp = pd.DataFrame(comp_rows)
            df_comp["OriginName"] = df_comp["OriginName"].astype(str).str.zfill(12)
            df_comp.to_csv(fr"{HI_OUT_DIR}\comp_{comp_id}.csv", index=False)

        except Exception as e:
            print(f"ERROR HI component {comp_id}: {e}")

    # HI: Summarize
    all_hi_files = glob.glob(fr"{HI_OUT_DIR}\comp_*.csv")
    df_hi_all = pd.concat([pd.read_csv(f) for f in all_hi_files], ignore_index=True)
    df_hi_all["OriginName"] = df_hi_all["OriginName"].astype(str).str.zfill(12)
    df_hi_all["layer"] = df_hi_all["method"].map({"road": 1, "multimodal": 2})

    df_hi_reachable = df_hi_all[df_hi_all["status"] != "unreachable"].copy()
    df_hi_min = (
        df_hi_reachable
        .sort_values("travel_time_min")
        .groupby("OriginName").first()
        .reset_index()
        [["OriginName", "DestinationName", "travel_time_min", "status", "method", "layer"]]
    )
    solved_hi_bg_ids = set(df_hi_min["OriginName"])
    unreachable_hi_bg_ids = [
        bg_id for bg_id in df_hi_all["OriginName"].unique()
        if bg_id not in solved_hi_bg_ids
    ]
    if unreachable_hi_bg_ids:
        df_hi_result = pd.concat([
            df_hi_min,
            pd.DataFrame({
                "OriginName": unreachable_hi_bg_ids,
                "DestinationName": pd.NA, "travel_time_min": pd.NA,
                "status": "unreachable", "method": pd.NA, "layer": pd.NA
            })
        ], ignore_index=True)
    else:
        df_hi_result = df_hi_min.copy()

    df_hi_result["NAICS_CODE"] = NAICS_CODE
    df_hi_result["OriginName"] = df_hi_result["OriginName"].astype(str).str.zfill(12)
    df_hi_result.to_parquet(fr"{PARQUET_DIR}\HI_result_{NAICS_CODE}.parquet", index=False)
    hi_all_results.append(df_hi_result)
    print(f"  HI done: {len(df_hi_result)} BGs | {df_hi_result['status'].value_counts().to_dict()}")

    print(f"[{NAICS_CODE}] Complete.")


# ============================================================
# Save combined parquet files
# ============================================================
df_ak_all_combined = pd.concat(ak_all_results, ignore_index=True)
df_ak_all_combined.to_parquet(fr"{PARQUET_DIR}\AK_result_all.parquet", index=False)
print(f"AK_result_all.parquet saved: {len(df_ak_all_combined)} rows")

df_hi_all_combined = pd.concat(hi_all_results, ignore_index=True)
df_hi_all_combined.to_parquet(fr"{PARQUET_DIR}\HI_result_all.parquet", index=False)
print(f"HI_result_all.parquet saved: {len(df_hi_all_combined)} rows")

print("\nAll done.")
