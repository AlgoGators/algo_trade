from ibapi import tag_value

class TagValue(tag_value.TagValue):    
    def __eq__(self, other : 'TagValue'):
        return self.tag == other.tag and self.value == other.value