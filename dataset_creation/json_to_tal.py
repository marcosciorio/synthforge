"""Convert a JSON file of parameters into a TAL-NoiseMaker preset.

Usage: python json_to_tal.py <preset.json>
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom


DEFAULT_PARAMS = {
    "unknown": "0.5",           
    "volume": "0.8",            
    "oscmastertune": "0.5",     
    "transpose": "0.5",         
    "portamento": "0.0",        
    "portamentomode": "0.0",
    "panic": "0.0",
    "voices": "0.0", 

    "ampattack": "0.0",
    "ampdecay": "0.0",
    "ampsustain": "1.0",        
    "amprelease": "0.0",
    "filterattack": "0.0",
    "filterdecay": "0.0",
    "filtersustain": "0.0",     
    "filterrelease": "0.0",
    "filtercontour": "0.5",     
    "keyfollow": "0.0",         
    "freeadattack": "0.0",
    "freeaddecay": "0.0",
    "freeadamount": "0.0",
    "freeaddestination": "0.0",

    "lfo1amount": "0.0",
    "lfo1rate": "0.0",
    "lfo1sync": "0.0",
    "lfo1waveform": "0.0",
    "lfo1keytrigger": "0.0",
    "lfo1destination": "0.0",
    "lfo1Phase": "0.0",
    "lfo2amount": "0.0",
    "lfo2rate": "0.0",
    "lfo2sync": "0.0",
    "lfo2waveform": "0.0",
    "lfo2keytrigger": "0.0",
    "lfo2destination": "0.0",
    "lfo2Phase": "0.0",

    "velocityvolume": "0.0",      
    "velocitycontour": "0.0",
    "velocityfilter": "0.0",
    "pitchwheelcutoff": "0.0",
    "pitchwheelpitch": "0.0",

    "chorus1enable": "0.0",
    "chorus2enable": "0.0",
    "reverbwet": "0.0",
    "delaywet": "0.0",
    "vintagenoise": "0.0",
    "highpass": "0.0",
    "detune": "0.0",
    "filterdrive": "0.0",

    "midilearn": "0.0",
    "midiclear": "0.0",
    "midilock": "0.0",
    "envelopeeditordest1": "0.0",
    "envelopeeditorspeed": "0.0",
    "envelopeeditoramount": "0.0",
    "envelopeOneShot": "0.0",
    "envelopefixtempo": "0.0",
    "envelopereset": "0.0"
}

GUI_NAME_FIXES = {
    "filtercutoff": "cutoff",
    "filterresonance": "resonance"
}

def create_preset(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error while loading the JSON file: {e}")
        return

    preset_name = os.path.splitext(os.path.basename(json_file))[0]
    
    final_params = DEFAULT_PARAMS.copy()
    final_params["programname"] = preset_name
    
    for key, value in data.items():
        xml_key = GUI_NAME_FIXES.get(key, key)
        final_params[xml_key] = str(round(float(value), 6))

    tal_root = ET.Element("tal", {
        "curprogram": "0",
        "version": "1.7",
        "presetName": preset_name,
        "path": f"User Presets/{preset_name}.noisemakerpreset"
    })
    
    programs = ET.SubElement(tal_root, "programs")
    program = ET.SubElement(programs, "program", final_params)
    
    spline_points = ET.SubElement(program, "splinePoints")
    ET.SubElement(spline_points, "splinePoint", {
        "isStartPoint": "1", "isEndPoint": "0", "centerPointX": "0.0", "centerPointY": "0.5",
        "controlPointLeftX": "0.0", "controlPointLeftY": "0.5", "controlPointRightX": "0.1000000014901161", "controlPointRightY": "0.5"
    })
    ET.SubElement(spline_points, "splinePoint", {
        "isStartPoint": "0", "isEndPoint": "1", "centerPointX": "1.0", "centerPointY": "0.5",
        "controlPointLeftX": "0.8999999761581421", "controlPointLeftY": "0.5", "controlPointRightX": "1.0", "controlPointRightY": "0.5"
    })
    
    ET.SubElement(tal_root, "midimap")

    xml_str = ET.tostring(tal_root, 'utf-8')
    reparsed = minidom.parseString(xml_str)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding="UTF-8").decode("UTF-8")
    
    output_filename = f"{preset_name}.noisemakerpreset"
    with open(output_filename, "w", encoding="UTF-8") as f:
        f.write(pretty_xml)
    
    print(f"✅ Preset written: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_tal.py <preset.json>")
    else:
        create_preset(sys.argv[1])