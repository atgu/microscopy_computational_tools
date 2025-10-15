import re
import xml.etree.ElementTree as ET

def get_index_xml(bucket_name, input_folder, plate):
    from google.cloud import storage
    storage_client = storage.Client()
    
    blobs = storage_client.list_blobs(bucket_name, prefix=f'{input_folder}/{plate}/Index')
    blobs = [blob for blob in blobs if blob.name.endswith('.xml')]
    if len(blobs) == 0:
        blobs = storage_client.list_blobs(bucket_name, prefix=f'{input_folder}/{plate}/Images/Index')
        blobs = [blob for blob in blobs if blob.name.endswith('.xml')]
    if len(blobs) == 0:
        return None
    return blobs[0].download_as_string()

def get_tree(xmlstring):
    xmlstring = xmlstring.decode('utf-8')
    xmlstring = re.sub(' xmlns="[^"]+"', '', xmlstring, count=1)
    return ET.ElementTree(ET.fromstring(xmlstring))

def get_channel_mapping(xmlstring):
    tree = get_tree(xmlstring)
    root = tree.getroot()
    channel_mapping = dict()
    
    for im in root.iter('Image'):
        channel_id = im.find('ChannelID')
        channel_name = im.find('ChannelName')
        if channel_id is not None and channel_name is not None:
            channel_id = channel_id.text
            channel_name = channel_name.text
            if channel_name in channel_mapping and channel_mapping[channel_name] != channel_id:
                return None
            channel_mapping[channel_name] = channel_id

    for entry in root.iter('Entry'):
        channel_id = entry.get('ChannelID')
        channel_name = entry.find('ChannelName')
        if channel_id is not None and channel_name is not None:
            channel_name = channel_name.text
            if channel_name in channel_mapping and channel_mapping[channel_name] != channel_id:
                return None
            channel_mapping[channel_name] = channel_id
    return channel_mapping
